#!/usr/bin/env python3
"""Enlarge text conservatively in frozen SVGs, without relaying out plots.

Worker only. The SVG viewBox and every non-text drawing element stay fixed.
System librsvg/cairo render PNG/PDF offline; no datasets or checkpoints load.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import ctypes as ct
import json
from pathlib import Path
import re
import socket
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
PAPER = ROOT / "scaling-csy/doc/paper_materials_20260728"
OUTPUT = PAPER / "figures/original_layout_font_tuning_20260917"
NS = "{http://www.w3.org/2000/svg}"
ET.register_namespace("", NS[1:-1])
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
SOURCES = {
    "single_to_single_all_domains": "single_to_single_publication_20260913",
    "multi_to_single_all_domains": "multi_to_single_publication_20260915",
    "single_to_all_overview": "cross_domain_publication_20260914",
    "three_frontiers_2x5": "frontier_comparison_2x5_20260915",
    "bpe_merge_hierarchy": "bpe_shared_hierarchy_20260915",
}


def require(value, message):
    if not value:
        raise ValueError(message)


class Rect(ct.Structure):
    _fields_ = [(key, ct.c_double) for key in ("x", "y", "width", "height")]


class GError(ct.Structure):
    _fields_ = [("domain", ct.c_uint32), ("code", ct.c_int), ("message", ct.c_char_p)]


class SVGRenderer:
    def __init__(self):
        self.rsvg = ct.CDLL("librsvg-2.so.2")
        self.cairo = ct.CDLL("libcairo.so.2")
        self.gobject = ct.CDLL("libgobject-2.0.so.0")
        self.glib = ct.CDLL("libglib-2.0.so.0")
        p = ct.c_void_p
        ep = ct.POINTER(ct.POINTER(GError))
        self.bind(self.rsvg, "rsvg_handle_new_from_data", p, [ct.c_char_p, ct.c_size_t, ep])
        self.bind(self.rsvg, "rsvg_handle_set_dpi", None, [p, ct.c_double])
        self.bind(self.rsvg, "rsvg_handle_get_geometry_for_layer", ct.c_int,
                  [p, ct.c_char_p, ct.POINTER(Rect), ct.POINTER(Rect), ct.POINTER(Rect), ep])
        self.bind(self.rsvg, "rsvg_handle_render_document", ct.c_int,
                  [p, p, ct.POINTER(Rect), ep])
        self.bind(self.cairo, "cairo_image_surface_create", p, [ct.c_int, ct.c_int, ct.c_int])
        self.bind(self.cairo, "cairo_pdf_surface_create", p, [ct.c_char_p, ct.c_double, ct.c_double])
        self.bind(self.cairo, "cairo_surface_write_to_png", ct.c_int, [p, ct.c_char_p])
        self.bind(self.cairo, "cairo_create", p, [p])
        self.bind(self.cairo, "cairo_scale", None, [p, ct.c_double, ct.c_double])
        self.bind(self.cairo, "cairo_destroy", None, [p])
        self.bind(self.cairo, "cairo_surface_finish", None, [p])
        self.bind(self.cairo, "cairo_surface_status", ct.c_int, [p])
        self.bind(self.cairo, "cairo_surface_destroy", None, [p])
        self.bind(self.gobject, "g_object_unref", None, [p])
        self.bind(self.glib, "g_error_free", None, [ct.POINTER(GError)])

    @staticmethod
    def bind(lib, name, result, args):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args

    def check(self, value, error, context):
        if not value:
            message = error.contents.message.decode() if error else "no diagnostic"
            if error:
                self.glib.g_error_free(error)
            raise RuntimeError(f"{context}: {message}")

    def handle(self, root):
        raw = ET.tostring(root, encoding="utf-8")
        error = ct.POINTER(GError)()
        handle = self.rsvg.rsvg_handle_new_from_data(raw, len(raw), ct.byref(error))
        self.check(handle, error, "SVG parse")
        self.rsvg.rsvg_handle_set_dpi(handle, 72)
        return handle

    @staticmethod
    def viewport(root):
        return Rect(*map(float, root.get("viewBox").split()))

    def boxes(self, root, texts):
        handle = self.handle(root)
        view = self.viewport(root)
        result = []
        try:
            for text in texts:
                ink, logical, error = Rect(), Rect(), ct.POINTER(GError)()
                ok = self.rsvg.rsvg_handle_get_geometry_for_layer(
                    handle, ("#" + text.get("id")).encode(), ct.byref(view),
                    ct.byref(ink), ct.byref(logical), ct.byref(error))
                self.check(ok, error, "SVG text geometry")
                result.append(np.array([ink.x, ink.y, ink.x + ink.width, ink.y + ink.height]))
        finally:
            self.gobject.g_object_unref(handle)
        return np.asarray(result)

    def export(self, root, stem, dpi=300):
        handle, view = self.handle(root), self.viewport(root)
        try:
            for extension in ("png", "pdf"):
                if extension == "png":
                    scale = dpi / 72
                    surface = self.cairo.cairo_image_surface_create(
                        0, round(view.width * scale), round(view.height * scale))
                else:
                    scale = 1
                    surface = self.cairo.cairo_pdf_surface_create(
                        str(stem.with_suffix(".pdf")).encode(), view.width, view.height)
                context = self.cairo.cairo_create(surface)
                try:
                    self.cairo.cairo_scale(context, scale, scale)
                    error = ct.POINTER(GError)()
                    ok = self.rsvg.rsvg_handle_render_document(handle, context, ct.byref(view), ct.byref(error))
                    self.check(ok, error, "SVG render")
                    if extension == "png":
                        status = self.cairo.cairo_surface_write_to_png(surface, str(stem.with_suffix(".png")).encode())
                        require(status == 0, f"PNG write failed: {status}")
                    self.cairo.cairo_surface_finish(surface)
                    require(self.cairo.cairo_surface_status(surface) == 0, "Cairo surface failed")
                finally:
                    self.cairo.cairo_destroy(context)
                    self.cairo.cairo_surface_destroy(surface)
        finally:
            self.gobject.g_object_unref(handle)


def matrix(raw):
    result = np.eye(3)
    for name, body in re.findall(r"(\w+)\(([^)]*)\)", raw or ""):
        args = [float(n) for n in re.split(r"[,\s]+", body.strip()) if n]
        m = np.eye(3)
        if name == "translate":
            m[:2, 2] = (args[0], args[1] if len(args) > 1 else 0)
        elif name == "rotate":
            angle = np.deg2rad(args[0])
            m[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            if len(args) == 3:
                anchor = np.array(args[1:])
                m[:2, 2] = anchor - m[:2, :2] @ anchor
        elif name == "scale":
            m[0, 0], m[1, 1] = args[0], args[-1]
        elif name == "matrix":
            a, b, c, d, e, f = args
            m = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        else:
            raise ValueError(f"Unsupported text transform {name}")
        result = result @ m
    return result


def text_info(root, stem):
    parents = {child: parent for parent in root.iter() for child in parent}
    texts = list(root.iter(NS + "text"))
    info = []
    for i, text in enumerate(texts):
        text.set("id", f"font-tuning-{i}")
        ancestors, parent = [], parents.get(text)
        while parent is not None:
            ancestors.append(parent)
            parent = parents.get(parent)
        ids = [n.get("id", "") for n in ancestors]
        contents = "".join(text.itertext()).strip()
        sizes = [float(s) for n in text.iter() for s in re.findall(r"font-size:\s*([\d.]+)px", n.get("style", ""))]
        require(sizes, f"No explicit font size for {contents}")
        role, factor = "annotation", 1.14
        if any(s.startswith("legend_") for s in ids):
            # Legend handles are fixed too: leave their original breathing room.
            role, factor = "legend", 1.08
            if stem == "single_to_single_all_domains" and len(contents) > 15:
                factor = 1.04
        elif any(s.startswith(("xtick_", "ytick_")) for s in ids):
            role, factor = "tick", 1.20
        elif any(s.startswith("matplotlib.axis_") for s in ids):
            role, factor = "axis_label", 1.10
        elif max(sizes) >= 10:
            role, factor = "heading", 1.06
        if stem == "bpe_merge_hierarchy":
            if contents in ("N", "C", "O", "P", "S"):
                role, factor = "atom_glyph", 1.0
            elif any("Mono" in n.get("style", "") for n in text.iter()):
                role, factor = "sequence", 1.10
                if any(s.startswith(("card-dna__", "card-rna__")) for s in ids):
                    factor = 1.0
            elif role != "legend":
                role, factor = "bpe_label", 1.12
        # Boxed coefficients and multiline notes already occupy tight spaces.
        if stem == "scaling_law_main" and role == "annotation":
            factor = 1.12
        transform = np.eye(3)
        for parent in reversed(ancestors):
            transform = transform @ matrix(parent.get("transform", ""))
        anchor = "left" if role == "legend" else "center"
        if role == "tick" and any(s.startswith("ytick_") for s in ids):
            anchor = "right"
        container = None
        for parent in ancestors:
            patches = [n for n in parent if n.tag == NS + "g" and n.get("id", "").startswith("patch_")]
            if patches and not parent.get("id", "").startswith(("axes_", "figure_")):
                container = patches[0].get("id")
                break
        info.append({"text": contents, "role": role, "requested_factor": factor,
                     "anchor": anchor, "parent_matrix": transform, "font_sizes": sizes})
        info[-1]["container"] = container
    return texts, info


def adjust_text(original, info, factors, renderer, old_boxes):
    root = copy.deepcopy(original)
    texts = list(root.iter(NS + "text"))
    for text, factor in zip(texts, factors):
        for node in text.iter():
            if "style" in node.attrib:
                node.set("style", re.sub(r"font-size:\s*([\d.]+)px",
                    lambda m: f"font-size: {float(m[1]) * factor:.6f}px", node.get("style")))
            if node.tag == NS + "tspan":
                for key in ("x", "y", "dx", "dy"):
                    if key in node.attrib:
                        node.set(key, " ".join(f"{float(v) * factor:.6f}" for v in node.get(key).split()))
    boxes = renderer.boxes(root, texts)
    for text, meta, old, new in zip(texts, info, old_boxes, boxes):
        side = meta["anchor"]
        dx = (old[0] - new[0]) if side == "left" else (old[2] - new[2]) if side == "right" else (old[0]+old[2]-new[0]-new[2])/2
        dy = (old[1]+old[3]-new[1]-new[3])/2
        shift = np.linalg.solve(meta["parent_matrix"][:2, :2], [dx, dy])
        transform = text.get("transform", "")
        text.set("transform", f"translate({shift[0]:.6f} {shift[1]:.6f}) {transform}".strip())
    return root, renderer.boxes(root, texts)


def overlaps(a, b, pad=0):
    return min(a[2], b[2]) - max(a[0], b[0]) > -pad and min(a[3], b[3]) - max(a[1], b[1]) > -pad


def separation(a, b):
    return max(a[0]-b[2], b[0]-a[2], a[1]-b[3], b[1]-a[3])


def nontext_snapshot(root):
    def visit(node):
        if node.tag == NS + "text":
            return ("TEXT",)
        return (node.tag, tuple(sorted(node.attrib.items())), node.text,
                tuple(visit(child) for child in node))
    return visit(root)


def restore_main_svg(output):
    # The original main figure has PDF/PNG only. Use its unchanged renderer and
    # frozen evidence, with its original canvas/style/layout, to obtain SVG.
    import plot_iclr_materials as evidence
    import matplotlib.pyplot as plt
    old, _, _, curves, metrics, fits = evidence.main_inputs()
    plt.rcdefaults()
    old.configure_style()
    plt.rcParams["svg.fonttype"] = "none"
    source = output / "runtime/scaling_law_main_original.svg"
    def capture(fig, out_dir, stem):
        fig.savefig(source, bbox_inches="tight")
        plt.close(fig)
    old.save_figure = capture
    ci = {key: dict(zip(("lower", "upper"), metrics[name]["bootstrap_95_ci"]))
          for key, name in (("a", "n_scaling"), ("b", "d_scaling"))}
    small = {**metrics["small_flagship_validation"], "L": metrics["small_flagship_validation"]["observed_loss"]}
    old.make_main_figure(curves, *fits, ci, metrics["a1_low_compute_validation"], small, output)
    return source


def tune(source, output, stem, renderer):
    original = ET.parse(source).getroot()
    texts, info = text_info(original, stem)
    old_boxes = renderer.boxes(original, texts)
    definitions = {node.get("id"): node for node in original.iter() if node.get("id")}
    container_ids = sorted({m["container"] for m in info if m["container"]})
    containers = dict(zip(container_ids, renderer.boxes(original, [definitions[k] for k in container_ids])))
    factors = np.array([m["requested_factor"] for m in info])
    _, _, width, height = map(float, original.get("viewBox").split())
    # Avoid new collisions. Already-touching text is not "fixed" by moving or
    # hiding it; just avoid making that pre-existing spacing tighter.
    pairs = [(i, j, min(1.75, separation(old_boxes[i], old_boxes[j]) * .70))
             for i in range(len(texts)) for j in range(i)
             if separation(old_boxes[i], old_boxes[j]) > .20]
    for iteration in range(20):
        revised, boxes = adjust_text(original, info, factors, renderer, old_boxes)
        crowded = set()
        for i, j, gap in pairs:
            if separation(boxes[i], boxes[j]) < gap:
                crowded.update((i, j))
        for i, box in enumerate(boxes):
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                crowded.add(i)
            if info[i]["container"]:
                boundary = containers[info[i]["container"]]
                old = old_boxes[i]
                if (box[0] < min(boundary[0]+.7, old[0])-.005 or
                    box[1] < min(boundary[1]+.7, old[1])-.005 or
                    box[2] > max(boundary[2]-.7, old[2])+.005 or
                    box[3] > max(boundary[3]-.7, old[3])+.005):
                    crowded.add(i)
        adjustable = {i for i in crowded if factors[i] > 1.000001}
        if not adjustable:
            require(not crowded, f"Original text/page already collides: {stem}, {crowded}")
            break
        for i in adjustable:
            factors[i] = max(1.0, factors[i] - .025)
    else:
        raise ValueError(f"Font spacing did not settle for {stem}")
    require(nontext_snapshot(original) == nontext_snapshot(revised), "Non-text geometry changed")
    require([list(t.itertext()) for t in original.iter(NS+"text")] ==
            [list(t.itertext()) for t in revised.iter(NS+"text")], "Text content changed")
    path = output / f"{stem}.svg"
    ET.ElementTree(revised).write(path, encoding="utf-8", xml_declaration=True)
    renderer.export(revised, output / stem)
    with Image.open(output / f"{stem}.png") as image:
        image.load()
        image.thumbnail((2100, 1800))
        image.convert("RGB").save(output / f"{stem}_preview.jpg", quality=93)
    report = {"source": str(source), "viewBox": original.get("viewBox"),
              "all_nontext_geometry_unchanged": True, "all_text_content_unchanged": True,
              "new_text_overlaps": 0, "font_factor_range": [float(min(factors)), float(max(factors))],
              "roles": dict(Counter(m["role"] for m in info)),
              "texts": [{k: v for k, v in m.items() if k != "parent_matrix"} |
                        {"applied_factor": round(float(f), 4)} for m, f in zip(info, factors)]}
    (output / f"{stem}_font_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print("FONT_ONLY_COMPLETE", stem, len(texts), report["font_factor_range"], flush=True)
    return {k: v for k, v in report.items() if k != "texts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figures", nargs="+", choices=[*SOURCES, "scaling_law_main"], default=[*SOURCES, "scaling_law_main"])
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Use the allocated CPU worker")
    (args.output / "runtime").mkdir(parents=True, exist_ok=True)
    renderer, reports = SVGRenderer(), {}
    for stem in args.figures:
        source = (restore_main_svg(args.output) if stem == "scaling_law_main" else
                  PAPER / "figures" / SOURCES[stem] / f"{stem}.svg")
        reports[stem] = tune(source, args.output, stem, renderer)
    if "bpe_merge_hierarchy" in args.figures:
        evidence = PAPER / "figures" / SOURCES["bpe_merge_hierarchy"] / "selected_merges.json"
        (args.output / evidence.name).write_bytes(evidence.read_bytes())
    (args.output / "font_only_audit.json").write_text(json.dumps(reports, indent=2)+"\n")
    print("FONT_ONLY_BATCH_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
