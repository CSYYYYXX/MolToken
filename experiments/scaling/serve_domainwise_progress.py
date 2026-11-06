#!/usr/bin/env python3
"""Serve only the exported live dashboard; no directory listing or file routes."""
from __future__ import annotations

import argparse
from datetime import datetime
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import traceback
from urllib.parse import urlsplit


class FrameDocument(HTMLParser):
    document = None

    def handle_starttag(self, tag, attrs):
        if tag == "iframe":
            self.document = dict(attrs).get("srcdoc")


def standalone_fragment(fragment, d3):
    """Preserve the inline source; replace the conversation-only refresh action."""
    start = fragment.index("      const update = root.querySelector('#holomol-update');")
    end = fragment.index("      if (!window.d3)", start)
    fragment = (fragment[:start]
                + "      root.querySelector('#holomol-update').remove();\n"
                + fragment[end:])
    external = '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>'
    if fragment.count(external) != 1 or "</script" in d3.lower():
        raise ValueError("Unexpected dashboard/library export contract")
    return fragment.replace(external, '<script>\n' + d3 + '\n</script>')


class LiveExport:
    def __init__(self, fragment, renderer, library, output):
        self.fragment, self.renderer, self.library, self.output = fragment, renderer, library, output
        self.lock = threading.Lock()
        self.version = None
        self.content = None

    def current(self):
        with self.lock:
            version = (self.fragment.stat().st_mtime_ns, self.library.stat().st_mtime_ns)
            if self.content is not None and version == self.version:
                return self.content
            prepared = self.output / "standalone-fragment.html"
            prepared.write_text(standalone_fragment(self.fragment.read_text(), self.library.read_text()), encoding="utf-8")
            exported = self.output / "exported-shell.html"
            subprocess.run([sys.executable, str(self.renderer), str(prepared), str(exported),
                            "--title", "HoloMol scaling progress"], check=True, stdout=subprocess.DEVNULL,
                           timeout=15)
            # Reuse the supplied export's full document/style/CSP, without an
            # iframe; native refresh must fetch the next NFS snapshot.
            parser = FrameDocument()
            parser.feed(exported.read_text(encoding="utf-8"))
            if not parser.document:
                raise ValueError("Export did not contain its expected frame document")
            # The dashboard has its own tooltips and no icons. The generic
            # export's optional tooltip/icon loaders are unused here.
            document = re.sub(r'<script src="https://unpkg\.com/[^\"]+"></script>', '', parser.document)
            document = document.replace('<head>', '<head>\n<meta http-equiv="refresh" content="60">', 1)
            document = document.replace('html>body{padding:0}', 'html>body{box-sizing:border-box;padding:16px;max-width:1100px;margin:0 auto}')
            document = document.replace('后台每 60 秒重绘 · 完成标记＋完整验证，非最终验收',
                                        '每 60 秒自动刷新 · 完成标记＋完整验证，非最终验收')
            temporary = self.output / "index.html.tmp"
            temporary.write_text(document, encoding="utf-8")
            os.replace(temporary, self.output / "index.html")
            self.content, self.version = document.encode("utf-8"), version
            return self.content


def make_handler(exporter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HoloMolDashboard/1"

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond(head=False)

        def respond(self, *, head):
            if urlsplit(self.path).path not in ("/", "/index.html"):
                self.send_error(404)
                return
            try:
                document = exporter.current()
            except Exception:
                traceback.print_exc()
                self.send_error(503, "Dashboard export unavailable; inspect server log")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(document)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head:
                self.wfile.write(document)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export = LiveExport(args.fragment, args.renderer, args.d3, args.output)
    export.current()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(export))
    identity = {"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(), "sid": os.getsid(0),
                "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                "host": args.host, "port": args.port, "status": "running",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (args.output / "server.pid.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"Listening on {args.host}:{args.port}; only / and /index.html; no directory exposure", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        identity["status"] = "stopped"
        (args.output / "server.pid.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
