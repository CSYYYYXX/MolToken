#!/usr/bin/env python3
"""Continuously publish validated results and all-observation bracketing plots.

The monitor never signals trainers or releases workers. Expected incomplete
JSONL appends are ignored until their newline arrives. Failed trainers remain
visible alerts while other experiments can continue producing results.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import socket
import time

from collect_domainwise_cluster_results import collect_results, read_plan_list_files, write_tables
from plot_domainwise_completed_snapshot import (
    DOMAINS, analyze_snapshot, load_snapshot, plot_curves, plot_domain_overlays, plot_peptide,
)
from run_domainwise_cluster_completion import COMPLETION_SCHEMA, atomic_json, load_json


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def progress_snapshot(plan_paths):
    by_id = {}
    for path in plan_paths:
        for row in load_json(path)["rows"]:
            by_id.setdefault(row["run_id"], []).append(row)
    progress = []
    for run_id,attempts in sorted(by_id.items()):
        row = max(attempts, key=lambda item:int(item["attempt_number"]))
        marker_path = Path(row["run_dir"])/"complete.json"
        if marker_path.exists():
            marker = load_json(marker_path)
            if marker.get("schema_version") != COMPLETION_SCHEMA or marker.get("run_id") != run_id:
                raise ValueError(f"completion marker identity mismatch for {run_id}")
            matching = [candidate for candidate in attempts
                        if Path(candidate["attempt_dir"]).resolve()==Path(marker["attempt_dir"]).resolve()]
            if not matching:
                raise ValueError(f"completion attempt is not in supplied plans for {run_id}")
            row = matching[-1]
        attempt = Path(row["attempt_dir"])
        item = {"run_id":run_id,"domain":row["domain"],"budget":row["budget"],
                "attempt_dir":str(attempt),"target_input_tokens":row["target_input_tokens_with_eos"]}
        if marker_path.exists():
            item["state"] = "complete"
        elif (attempt/"failed.json").exists():
            item.update(state="failed",failure=load_json(attempt/"failed.json"))
        else:
            item["state"] = "not_started"
            log = attempt/"train_log.jsonl"
            if log.exists():
                with log.open("rb") as stream:
                    size = stream.seek(0,2)
                    offset = max(0,size-65536)
                    stream.seek(offset)
                    lines = stream.read().splitlines(keepends=True)
                if offset:
                    lines = lines[1:]
                events = [json.loads(line) for line in lines if line.endswith(b"\n")]
                updates = [e for e in events if e.get("event")=="train"]
                age = max(0,time.time()-log.stat().st_mtime)
                item.update(log_age_seconds=age,state="initializing_or_final_validation")
                if updates:
                    last = updates[-1]
                    tokens = int(last["cumulative_input_tokens"])
                    item.update(state="recent_training_log" if age<300 else "stale_log_requires_process_check",
                                step=last["step"],input_tokens=tokens,
                                token_fraction=tokens/int(row["target_input_tokens_with_eos"]),loss=last["loss"])
                    if len(updates)>1:
                        elapsed = last["elapsed_sec"]-updates[0]["elapsed_sec"]
                        if elapsed>0:
                            rate=(tokens-updates[0]["cumulative_input_tokens"])/elapsed
                            if rate>0:
                                item.update(recent_input_tokens_per_sec=rate,
                                            estimated_train_hours_remaining=max(0,row["target_input_tokens_with_eos"]-tokens)/rate/3600)
                    if tokens>=int(row["target_input_tokens_with_eos"]):
                        item["state"]="training_budget_reached_waiting_final_validation"
        progress.append(item)
    return progress


def publish_page(output, status):
    body=["<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='60'>",
          "<title>HoloMol 实时分领域 scaling</title>",
          "<style>body{font:16px system-ui;max-width:1500px;margin:25px auto;padding:20px}img{width:100%}td,th{padding:6px 12px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>",
          "<h1>HoloMol · 分领域 Scaling 实时结果</h1>",
          "<p id='heartbeat'></p>",
          f"<p>更新：{html.escape(status['updated_at'])} · 完整 validation：{status['completed_runs']} 点 · 待完成：{status['pending_runs']} 点 · 已包夹：{status['accepted_curves']}/24 条标准领域曲线</p>",
          "<p>图中只使用完整 validation；训练中间 loss 不参与拟合。新补点与旧点一起拟合，不选子集。多肽单独按等数据量展示。</p>",
          "<p><a href='domainwise_overlay.png'>总览 PNG</a> · <a href='domainwise_overlay.pdf'>PDF</a> · <a href='all_isoflop_curves.png'>逐预算放大图</a> · <a href='bracketing.json'>包夹诊断 JSON</a></p>",
          "<img src='domainwise_overlay.svg'>",
          "<h2>当前训练进度（不是拟合结果）</h2><table><tr><th>Run</th><th>状态</th><th>Token 进度</th><th>剩余训练小时估计</th></tr>"]
    for row in status["progress"]:
        fraction=f"{100*row['token_fraction']:.1f}%" if "token_fraction" in row else "—"
        eta=f"{row['estimated_train_hours_remaining']:.2f}" if "estimated_train_hours_remaining" in row else "—"
        body.append(f"<tr><td>{html.escape(row['run_id'])}</td><td>{html.escape(row['state'])}</td><td>{fraction}</td><td>{eta}</td></tr>")
    body.append("</table><p>ETA 仅按近期速度估计，不含完整 final validation。日志超时不代表进程已经停止，不能据此重启。</p>")
    body.append("<script>const updated="+json.dumps(status['updated_at'])+";const age=(Date.now()-Date.parse(updated))/1000;if(age>180){const p=document.getElementById('heartbeat');p.style.color='#b91c1c';p.textContent='监控心跳已超过 3 分钟未更新；这是旧快照，请核对 watcher 进程。';}</script>")
    temporary=output/"index.html.tmp"
    temporary.write_text("\n".join(body),encoding="utf-8")
    os.replace(temporary,output/"index.html")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tables",type=Path,required=True)
    parser.add_argument("--plan-list-file",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--publish-overlays-to",type=Path)
    parser.add_argument("--interval-seconds",type=float,default=60)
    parser.add_argument("--once",action="store_true")
    return parser.parse_args()


def run_watcher(args):
    if args.interval_seconds<30:
        raise ValueError("monitor interval must be at least 30 seconds")
    output=args.output_dir.absolute()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".watcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        process={"pid":os.getpid(),"hostname":socket.gethostname(),"started_at":now(),"state":"running"}
        atomic_json(output/"watcher_process.json",process)
        previous_signature=None
        while True:
            plans=read_plan_list_files([args.plan_list_file])
            payload=collect_results(baseline_completed_csv=args.baseline_tables/"completed_runs.csv",
                                    baseline_pending_csv=args.baseline_tables/"pending_runs.csv",plan_jsons=plans)
            write_tables(payload,output/"tables")
            done,pending=load_snapshot(output/"tables")
            curves=analyze_snapshot(done,pending)
            atomic_json(output/"bracketing.json",{"updated_at":now(),"metric_split":"validation",
                        "test_accessed":False,"curves":curves,"method":"all_observations_fixed_original_thresholds"})
            signature=(tuple(sorted((p["run_id"],p["validation_loss"]) for p in done)),tuple(sorted(p["run_id"] for p in pending)))
            if signature!=previous_signature:
                plot_domain_overlays(curves,done,output,snapshot_label="原机＋集群完整 validation")
                plot_curves(curves,output,snapshot_label="physical + cluster complete validation")
                plot_peptide(done,output)
                if args.publish_overlays_to:
                    gallery=args.publish_overlays_to
                    gallery.mkdir(parents=True,exist_ok=True)
                    names=["domainwise_overlay",*(f"{d}_overlay" for d in DOMAINS)]
                    for name in names:
                        for ext in ("png","svg","pdf"):
                            target=gallery/f"{name}.{ext}"
                            # Preserve the original 106-point visual snapshot once.
                            backup=gallery/"snapshot_106"/target.name
                            if target.exists() and not backup.exists():
                                backup.parent.mkdir(parents=True,exist_ok=True)
                                shutil.copy2(target,backup)
                            temp=target.with_suffix(target.suffix+".tmp")
                            shutil.copyfile(output/target.name,temp)
                            os.replace(temp,target)
                previous_signature=signature
            progress=progress_snapshot(plans)
            status={"updated_at":now(),"pid":os.getpid(),"hostname":socket.gethostname(),
                    "completed_runs":len(done),"pending_runs":len(pending),
                    "accepted_curves":sum(c["visual_category"]=="accepted" for c in curves),
                    "all_standard_curve_minima_bracketed":all(c["visual_category"]=="accepted" for c in curves),
                    "worker_release_authorized":False,"progress":progress}
            atomic_json(output/"status.json",status)
            atomic_json(output/"watcher_process.json",{**process,"last_update_at":status["updated_at"]})
            publish_page(output,status)
            if args.publish_overlays_to:
                gallery=args.publish_overlays_to
                destination=html.escape(os.path.relpath(output/"index.html",gallery),quote=True)
                temporary=gallery/"index.html.tmp"
                temporary.write_text("<!doctype html><meta charset='utf-8'>"
                                     +f"<meta http-equiv='refresh' content='0;url={destination}'>"
                                     +f"<h1><a href='{destination}'>打开实时结果页</a></h1>"
                                     +"<p>总览图持续更新；原机 106 点总览备份位于 snapshot_106/。</p>",encoding="utf-8")
                os.replace(temporary,gallery/"index.html")
            failures=[p["run_id"] for p in progress if p["state"]=="failed"]
            print(json.dumps({"updated_at":status["updated_at"],"completed":len(done),"pending":len(pending),
                              "accepted_curves":status["accepted_curves"],"failed_runs":failures}),flush=True)
            if args.once:
                atomic_json(output/"watcher_process.json",{**process,"state":"completed_once","last_update_at":now()})
                return
            time.sleep(args.interval_seconds)


def record_watcher_failure(output, error):
    failure={"pid":os.getpid(),"hostname":socket.gethostname(),"failed_at":now(),
             "state":"failed","error_type":type(error).__name__,"error":str(error)}
    atomic_json(output/"watcher_failed.json",failure)
    atomic_json(output/"watcher_process.json",failure)
    temporary=output/"index.html.tmp"
    temporary.write_text("<!doctype html><meta charset='utf-8'><h1>结果监控已停止：需要处理异常</h1>"
                         +"<pre>"+html.escape(json.dumps(failure,ensure_ascii=False,indent=2))+"</pre>"
                         +"<p>最后成功发布的图表仍保留，但不是当前进度。训练进程没有被此监控器停止。</p>"
                         +"<a href='domainwise_overlay.png'>最后成功发布的总览图</a>",encoding="utf-8")
    os.replace(temporary,output/"index.html")


def main():
    args=parse_args()
    try:
        run_watcher(args)
    except BlockingIOError:
        # Another writer owns this directory; do not overwrite its live status.
        raise
    except Exception as error:
        # Publish the failure, then retain the original traceback/non-zero exit.
        record_watcher_failure(args.output_dir.absolute(),error)
        raise


if __name__=="__main__":
    main()
