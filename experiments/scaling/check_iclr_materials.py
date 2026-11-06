#!/usr/bin/env python3
"""Worker-only font/page and PNG/PDF artifact checks for the ICLR re-layout."""
import argparse
import json
from pathlib import Path
import re
import socket
from PIL import Image

import plot_iclr_materials as plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--directory", type=Path, default=plots.OUTPUT)
    args = parser.parse_args()
    assert socket.gethostname() == args.expected_host and "worker" in args.expected_host
    results = {}
    for pdf in sorted(args.directory.glob("*.pdf")):
        stem = pdf.stem
        raw = pdf.read_bytes()
        box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
        assert box, f"No PDF page dimensions: {pdf}"
        x0, y0, x1, y1 = map(float, box.groups())
        width, height = (x1-x0)/72, (y1-y0)/72
        expected_width = 8.5 if stem == "bpe_merge_hierarchy" else 5.5
        assert abs(width-expected_width) < 1e-7, (stem, width)
        svg = pdf.with_suffix(".svg").read_text()
        assert "font-family" in svg and "<text" in svg, "SVG text must remain editable"
        with Image.open(pdf.with_suffix(".png")) as image:
            image.load()
            assert abs(image.width/350-width) < .01 and abs(image.height/350-height) < .01
            # Same visual content at screen-preview resolution; not a data edit.
            preview = image.convert("RGB")
            preview.thumbnail((1100, 1500))
            preview.save(args.directory / "runtime" / f"{stem}_preview.png")
            results[stem] = {"page_inches": [width, height], "raster_pixels": list(image.size),
                             "png_decoded": True, "editable_svg_text": True}
    assert set(results) == {"scaling_law_main", "single_to_single_all_domains", "multi_to_single_all_domains",
                            "single_to_all_overview", "three_frontiers_2x5", "bpe_merge_hierarchy"}, "Expected all six current figures"
    manifest = json.loads((args.directory / "rendering.json").read_text())
    assert set(manifest["figures"]) == set(results), "Incomplete rendering manifest"
    assert manifest["data_or_fit_changes"] is False and manifest["original_files_overwritten"] is False
    for name, figure in manifest["figures"].items():
        assert figure["nominal_text_pt_range"][0] >= 9, (name, "small font")
    (args.directory / "artifact_checks.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
