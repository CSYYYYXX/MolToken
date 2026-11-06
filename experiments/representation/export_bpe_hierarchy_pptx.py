#!/usr/bin/env python3
"""Convert the approved BPE SVG to native, grouped, editable PowerPoint shapes.

Read the SVG and its small provenance JSON only; no corpus or model loading.
This deliberately supports only the SVG constructs emitted by this figure.
Unknown constructs fail, rather than being rasterized or silently omitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import socket
import xml.etree.ElementTree as ET
import zipfile

from matplotlib.font_manager import FontProperties, findfont
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[5]
DEFAULT = ROOT / "scaling-csy/doc/paper_materials_20260728/figures/bpe_merge_hierarchy_20260914"
SVG_NS = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}href"
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


def require(value, message):
    if not value:
        raise ValueError(message)


def element(tag, **attributes):
    result = OxmlElement(tag)
    for key, value in attributes.items():
        result.set(key, str(value))
    return result


def style_dict(node, inherited):
    result = dict(inherited)
    for item in node.get("style", "").split(";"):
        if item.strip():
            key, value = item.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def translated(node, offset):
    """The source uses translations and zero-degree text rotations only."""
    x, y = offset
    raw = node.get("transform", "")
    matches = list(re.finditer(r"([A-Za-z]+)\(([^)]*)\)", raw))
    require(not re.sub(r"([A-Za-z]+)\(([^)]*)\)", "", raw).strip(), f"Unknown transform: {raw}")
    for match in matches:
        name, body = match.groups()
        numbers = [float(n) for n in re.findall(NUMBER, body)]
        if name == "translate":
            require(len(numbers) in (1, 2), f"Bad translation: {raw}")
            x += numbers[0]
            y += numbers[1] if len(numbers) == 2 else 0
        elif name == "rotate":
            require(numbers and numbers[0] == 0, f"Nonzero rotation unsupported: {raw}")
        else:
            raise ValueError(f"Unsupported SVG transform: {raw}")
    return x, y


def parse_path(raw, offset):
    tokens = re.findall(r"[MLQCZmlqcz]|" + NUMBER, raw)
    require(not re.sub(r"[MLQCZmlqcz]|" + NUMBER + r"|[\s,]", "", raw), "Unsupported SVG path command")
    commands, i = [], 0
    while i < len(tokens):
        command = tokens[i]
        i += 1
        require(command in ("M", "L", "Q", "C", "z", "Z"), f"Expected absolute command, got {command}")
        size = {"M": 2, "L": 2, "Q": 4, "C": 6, "z": 0, "Z": 0}[command]
        points = []
        for j in range(0, size, 2):
            points.append((float(tokens[i + j]) + offset[0], float(tokens[i + j + 1]) + offset[1]))
        i += size
        commands.append((command.upper(), points))
    require(commands and commands[0][0] == "M", "Path must start with moveTo")
    return commands


@lru_cache(maxsize=128)
def font_metrics(family, size, weight, italic):
    prop = FontProperties(family=family, weight=weight, style="oblique" if italic else "normal")
    font_path = findfont(prop, fallback_to_default=False)
    font = ImageFont.truetype(font_path, round(size * 100))
    ascent, descent = font.getmetrics()
    return font, ascent / 100, descent / 100


@dataclass
class Item:
    kind: str
    style: dict
    ancestors: tuple[str, ...]
    path: list | None = None
    text: str = ""
    x: float = 0
    y: float = 0
    family: str = ""
    size: float = 0
    weight: int = 400
    italic: bool = False
    box: tuple = ()


def make_text(text, x, y, style, ancestors):
    family = style.get("font-family", "'DejaVu Sans'").strip("'\"")
    size = float(style.get("font-size", "8px").removesuffix("px"))
    weight = int(style.get("font-weight", "400"))
    italic = style.get("font-style", "normal") in ("italic", "oblique")
    font, ascent, descent = font_metrics(family, size, weight, italic)
    advance = font.getlength(text) / 100
    anchor = style.get("text-anchor", "start")
    require(anchor in ("start", "middle", "end"), f"Unknown text anchor: {anchor}")
    left = x - (advance / 2 if anchor == "middle" else advance if anchor == "end" else 0)
    return Item("text", style, ancestors, text=text, x=x, y=y, family=family, size=size,
                weight=weight, italic=italic, box=(left, y - ascent, max(advance + .5, 1), ascent + descent))


def load_svg(source):
    root = ET.parse(source).getroot()
    view = [float(v) for v in root.get("viewBox").split()]
    require(view[:2] == [0, 0], "Expected a zero-origin SVG")
    definitions = {n.get("id"): n for n in root.iter() if n.get("id")}
    items = []

    def visit(node, inherited, offset, ancestors):
        tag = node.tag.removeprefix(SVG_NS)
        if tag in ("defs", "metadata"):
            return
        style = style_dict(node, inherited)
        offset = translated(node, offset)
        ancestors = ancestors + ((node.get("id"),) if node.get("id") else ())
        if tag in ("svg", "g"):
            for child in node:
                visit(child, style, offset, ancestors)
        elif tag == "path":
            commands = parse_path(node.get("d"), offset)
            points = [point for _, group in commands for point in group]
            xs, ys = zip(*points)
            box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            if ancestors[-1:] == ("patch_1",):
                require(box == (0, 0, view[2], view[3]) and style.get("fill") == "#ffffff",
                        "Only the white page background may be replaced by slide background")
                return
            items.append(Item("path", style, ancestors, path=commands, box=box))
        elif tag == "use":
            ref = node.get(XLINK)
            require(ref and ref.startswith("#") and ref[1:] in definitions, f"Missing SVG definition {ref}")
            referenced = definitions[ref[1:]]
            require(referenced.tag == SVG_NS + "path", "Only path definitions are supported")
            merged = dict(referenced.attrib)
            merged.pop("id", None)
            merged["style"] = ";".join(f"{key}:{value}" for key, value in style.items())
            proxy = ET.Element(referenced.tag, merged)
            visit(proxy, style, (offset[0] + float(node.get("x", "0")), offset[1] + float(node.get("y", "0"))), ancestors)
        elif tag == "text":
            if not list(node):
                items.append(make_text(node.text or "", float(node.get("x", "0")) + offset[0],
                                       float(node.get("y", "0")) + offset[1], style, ancestors))
            else:
                spans = []
                for child in node:
                    require(child.tag == SVG_NS + "tspan", "Unexpected child in SVG text")
                    substyle = style_dict(child, style)
                    x, y = translated(child, offset)
                    spans.append(make_text(child.text or "", x + float(child.get("x", "0")),
                                           y + float(child.get("y", "0")), substyle, ancestors))
                # Keep each numeric subscript editable as one number, rather than separate digits.
                combined = []
                for span in sorted(spans, key=lambda value: value.x):
                    previous = combined[-1] if combined else None
                    if (previous and previous.text.isdigit() and span.text.isdigit()
                            and previous.style == span.style and abs(previous.y - span.y) < .001):
                        font, _, _ = font_metrics(previous.family, previous.size, previous.weight, previous.italic)
                        end = previous.x + font.getlength(previous.text) / 100
                        require(abs(span.x - end) < .4, "Unexpected non-contiguous numeric subscript")
                        combined[-1] = make_text(previous.text + span.text, previous.x, previous.y, previous.style, ancestors)
                    else:
                        combined.append(span)
                items.extend(combined)
        else:
            raise ValueError(f"Unsupported visible SVG element: {tag}")

    visit(root, {"fill": "#000000", "stroke": "none", "stroke-linejoin": "round", "stroke-linecap": "butt"},
          (0, 0), ())
    return view[2:], items


def paint_shape(shape, style):
    fill = style.get("fill", "#000000")
    if fill == "none":
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill.lstrip("#"))
    stroke = style.get("stroke", "none")
    if stroke == "none":
        shape.line.fill.background()
        return
    shape.line.color.rgb = RGBColor.from_string(stroke.lstrip("#"))
    width = float(style.get("stroke-width", "1"))
    shape.line.width = Pt(width)
    ln = shape.line._get_or_add_ln()
    ln.set("cap", {"butt": "flat", "round": "rnd", "square": "sq"}[style.get("stroke-linecap", "butt")])
    dash = style.get("stroke-dasharray")
    if dash and dash != "none":
        pattern = [float(value) for value in re.split(r"[,\s]+", dash)]
        require(len(pattern) % 2 == 0 and width > 0, "Unsupported dash pattern")
        custom = element("a:custDash")
        for distance, gap in zip(pattern[::2], pattern[1::2]):
            custom.append(element("a:ds", d=round(distance / width * 100000), sp=round(gap / width * 100000)))
        ln.append(custom)
    join = style.get("stroke-linejoin", "round")
    ln.append(element("a:round") if join == "round" else element("a:miter", lim=800000))


def add_path(slide, item, number):
    left, top, width, height = item.box
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Pt(left), Pt(top), max(Pt(width), 1), max(Pt(height), 1))
    shape.name = f"Vector {number}: {item.ancestors[-1]}"
    sp = shape._element.spPr
    sp.remove(sp.prstGeom)
    geometry = element("a:custGeom")
    for tag in ("a:avLst", "a:gdLst", "a:ahLst", "a:cxnLst"):
        geometry.append(element(tag))
    geometry.append(element("a:rect", l=0, t=0, r="w", b="h"))
    paths = element("a:pathLst")
    path = element("a:path", w=shape.width, h=shape.height, fill="norm", stroke="1", extrusionOk="0")
    names = {"M": "a:moveTo", "L": "a:lnTo", "Q": "a:quadBezTo", "C": "a:cubicBezTo", "Z": "a:close"}
    for command, points in item.path:
        part = element(names[command])
        for x, y in points:
            part.append(element("a:pt", x=Pt(x - left), y=Pt(y - top)))
        path.append(part)
    paths.append(path)
    geometry.append(paths)
    sp.append(geometry)
    paint_shape(shape, item.style)
    return shape


def add_text(slide, item, number):
    left, top, width, height = item.box
    shape = slide.shapes.add_textbox(Pt(left), Pt(top), Pt(width), Pt(height))
    shape.name = f"Text {number}: {item.text}"
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.word_wrap = False
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame._txBody.bodyPr.set("vertOverflow", "overflow")
    paragraph = frame.paragraphs[0]
    paragraph.space_before = paragraph.space_after = Pt(0)
    paragraph.line_spacing = Pt(height)
    run = paragraph.add_run()
    run.text = item.text
    run.font.name = item.family
    run.font.size = Pt(item.size)
    run.font.bold = item.weight >= 600
    run.font.italic = item.italic
    run.font.color.rgb = RGBColor.from_string(item.style.get("fill", "#000000").lstrip("#"))
    run._r.get_or_add_rPr().set("kern", "0")
    return shape


def inside(inner, outer, tolerance=.8):
    x, y, w, h = inner
    a, b, c, d = outer
    return a - tolerance <= x and b - tolerance <= y and x + w <= a + c + tolerance and y + h <= b + d + tolerance


def group_shapes(slide, built, provenance):
    if "layout_groups" in provenance:
        names = {group["id"]: group["name"] for group in provenance["layout_groups"]}
        buckets = defaultdict(list)
        for item, shape in built:
            source_group = next((ancestor.split("__")[0] for ancestor in item.ancestors
                                 if ancestor.split("__")[0] in names), None)
            if source_group is None:
                source_group = next((a for a in item.ancestors if a.startswith(("legend_", "annotation-"))), None)
            require(source_group is not None, f"Ungrouped shared-hierarchy element: {item.ancestors}")
            buckets[source_group].append(shape)
        require(set(names) <= set(buckets), "A declared illustration group is empty")
        result = []
        for key, shapes in buckets.items():
            group = slide.shapes.add_group_shape(shapes)
            group.name = names.get(key, key)
            result.append(group.name)
        return result
    cards = sorted((item.box for item, _ in built if item.kind == "path" and item.style.get("stroke") == "#d3dce4"),
                   key=lambda box: (round(box[1]), box[0]))
    require(len(cards) == 12, f"Expected 12 merge cards, got {len(cards)}")
    buckets = defaultdict(list)
    paths = list(provenance["paths"].items())
    for item, shape in built:
        assigned = False
        for index, box in enumerate(cards):
            if inside(item.box, box):
                lane, ids = paths[index // 4]
                buckets[f"{lane} / Merge {index % 4 + 1} / T{ids[index % 4 + 1]}"].append(shape)
                assigned = True
                break
        if assigned:
            continue
        if "legend_1" in item.ancestors or "legend_2" in item.ancestors or "legend_3" in item.ancestors:
            key = next(a for a in item.ancestors if a.startswith("legend_"))
        elif inside(item.box, (19.44, 187.2, 109.44, 110.88)):
            key = "Base token T42"
        elif item.kind == "path":
            key = "Arrow " + next(a for a in reversed(item.ancestors) if a.startswith("patch_"))
        else:
            key = next((a for a in reversed(item.ancestors) if a.startswith("text_")), "Labels")
        buckets[key].append(shape)
    for name, shapes in buckets.items():
        group = slide.shapes.add_group_shape(shapes)
        group.name = name
    return list(buckets)


def walk_shapes(shapes):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk_shapes(shape.shapes)
        else:
            yield shape


def validate(destination, dimensions, items, groups, provenance):
    reopened = Presentation(destination)
    require(len(reopened.slides) == 1, "Expected one editable slide")
    require(reopened.slide_width == Pt(dimensions[0]) and reopened.slide_height == Pt(dimensions[1]), "Slide size changed")
    shapes = list(walk_shapes(reopened.slides[0].shapes))
    require(len(shapes) == len(items), "Some source elements were lost or duplicated")
    expected_text = Counter(item.text for item in items if item.kind == "text")
    actual_text = Counter(shape.text for shape in shapes if shape.has_text_frame and shape.text)
    require(actual_text == expected_text, "Editable text changed during export")
    with zipfile.ZipFile(destination) as archive:
        require(not any(name.startswith("ppt/media/") for name in archive.namelist()), "Raster/vector image was embedded instead of native shapes")
        require(archive.testzip() is None, "Corrupt PowerPoint container")
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    shape_ids = [node.get("id") for node in slide.findall(".//p:cNvPr", ns)]
    require(len(shape_ids) == len(set(shape_ids)), "Duplicate PowerPoint shape IDs")
    expected_paths = sum(item.kind == "path" for item in items)
    require(len(slide.findall(".//a:custGeom", ns)) == expected_paths, "Native vector geometry count mismatch")
    require(len(slide.findall(".//p:pic", ns)) == 0, "Picture element found")
    if "layout_groups" in provenance:
        cards = [g for g in provenance["layout_groups"] if g["id"].startswith("card-")]
        for card in cards:
            require(card["name"] in groups, "Missing shared-hierarchy card group")
        card_count = len(cards)
    else:
        for lane, path in provenance["paths"].items():
            for step, token in enumerate(path[1:], 1):
                require(f"{lane} / Merge {step} / T{token}" in groups, "Missing editable card group")
        card_count = 12
    return {"slide_count": 1, "slide_inches": [d / 72 for d in dimensions],
            "native_vector_shapes": expected_paths, "editable_text_boxes": sum(expected_text.values()),
            "top_level_groups": len(groups), "merge_card_groups": card_count, "embedded_images": 0,
            "text_preserved": True, "source": "Approved SVG geometry and editable text",
            "fonts": sorted({item.family for item in items if item.kind == "text"}),
            "office_application_render_test": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    require("worker" in args.expected_host and socket.gethostname() == args.expected_host, "Run on the confirmed CPU worker")
    source = args.directory / "bpe_merge_hierarchy.svg"
    destination = args.directory / "bpe_merge_hierarchy_editable.pptx"
    provenance = json.loads((args.directory / "selected_merges.json").read_text())
    dimensions, items = load_svg(source)
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = (Pt(value) for value in dimensions)
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    built = [(item, add_path(slide, item, index) if item.kind == "path" else add_text(slide, item, index))
             for index, item in enumerate(items, 1)]
    groups = group_shapes(slide, built, provenance)
    slide.notes_slide.notes_text_frame.text = (
        "BPE merge hierarchy — editable version of the approved figure.\n"
        "图形和文本均为 PowerPoint 原生元素，每个小图已分组；可双击组内元素或取消组合编辑。\n"
        "数字上下标是独立可编辑文本，保持源 SVG 的字体、字号和位置。\n"
        "字体：" + " / ".join(sorted({item.family for item in items if item.kind == "text"})) +
        "。若本机缺失，PowerPoint 可能替换字体。\n"
        "没有修改分子结构、序列、token ID、合并规则、布局或图例位置。\n"
        "Source: bpe_merge_hierarchy.svg; provenance: selected_merges.json."
    )
    presentation.core_properties.title = "BPE merge hierarchy — editable"
    presentation.core_properties.subject = "Native editable molecular graphs and token merge hierarchy"
    presentation.save(destination)
    report = validate(destination, dimensions, items, groups, provenance)
    (args.directory / "pptx_editability_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PPTX_COMPLETE", destination, json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
