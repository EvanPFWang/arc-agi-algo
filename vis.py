
import argparse, json, os, re
import numpy as np
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

#ARC 10-color palette
PALETTE = np.array([
    [0,   0,   0],      #0 black
    [0,   0,   255],    #1 blue
    [255, 0,   0],      #2 red
    [0,   255, 0],      #3 green
    [255, 255, 0],      #4 yellow
    [128, 128, 128],    #5 gray
    [255, 0,   255],    #6 magenta
    [255, 165, 0],      #7 orange
    [0,   255, 255],    #8 cyan
    [165, 42,  42],     #9 brown
], dtype=np.uint8)

def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def grid_to_rgb(grid: List[List[int]], scale: int = 20, draw_grid: bool = False) -> Image.Image:
    """Convert a 2D list of ints 0..9 to a scaled RGB PIL.Image."""
    arr = np.array(grid, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError("Grid must be 2D")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError("Grid values must be in 0..9")
    rgb = PALETTE[arr]
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    if draw_grid and scale >= 4:
        #draw subtle 1px grid lines
        draw = ImageDraw.Draw(img)
        for x in range(0, img.width + 1, scale):
            draw.line([(x, 0), (x, img.height-1)], fill=(40, 40, 40))
        for y in range(0, img.height + 1, scale):
            draw.line([(0, y), (img.width-1, y)], fill=(40, 40, 40))
    return img

def banner(img: Image.Image, text: str) -> Image.Image:
    """Add a small banner with text above an image."""
    pad = 6
    bar_h = 22
    W = img.width
    out = Image.new("RGB", (W, bar_h + pad + img.height), (255, 255, 255))
    d = ImageDraw.Draw(out)
    #banner background
    d.rectangle([0, 0, W, bar_h], fill=(245, 245, 245))
    #text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    safe = text.encode("ascii","replace").decode("ascii")
    d.text((6, 4), safe, fill=(30,30,30), font=font)
    out.paste(img, (0, bar_h + pad))
    return out

def hstack(images: List[Image.Image], gap: int = 8, bg=(255, 255, 255)) -> Image.Image:
    """Horizontally stack images with a gap; aligns to top."""
    if not images:
        raise ValueError("No images to stack")
    H = max(im.height for im in images)
    W = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out

def vstack(images: List[Image.Image], gap: int = 10, bg=(255, 255, 255)) -> Image.Image:
    """Vertically stack images with a gap; centers each row."""
    if not images:
        raise ValueError("No images to stack")
    W = max(im.width for im in images)
    H = sum(im.height for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (W, H), bg)
    y = 0
    for im in images:
        x = (W - im.width) // 2
        out.paste(im, (x, y))
        y += im.height + gap
    return out

def detect_format(obj: Dict[str, Any]) -> str:
    """
    Return "challenges" if values look like {"train":[...], "test":[...]}.
    Return "solutions" if values look like a list of grids (test outputs).
    """
    if not obj:
        raise ValueError("Empty JSON")
    sample = next(iter(obj.values()))
    if isinstance(sample, dict) and "train" in sample and "test" in sample:
        return "challenges"
    if isinstance(sample, list):
        #solutions: list of 2D lists
        return "solutions"
    raise ValueError("Unknown ARC JSON structure")

def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def export_solutions_only(solutions: Dict[str, List[List[List[int]]]], outdir: Path, scale: int, draw_grid: bool, make_composite: bool):
    """Export test outputs only."""
    for tid, outs in solutions.items():
        tdir = outdir / sanitize(tid)
        ensure_dir(tdir)
        row_imgs = []
        for j, grid in enumerate(outs, start=1):
            im = grid_to_rgb(grid, scale=scale, draw_grid=draw_grid)
            im_b = banner(im, f"{tid} • test#{j} • output")
            im_path = tdir / f"{sanitize(tid)}__test{j}_output.png"
            im_b.save(im_path)
            row_imgs.append(im_b)
        if make_composite and row_imgs:
            (tdir / "composite").mkdir(exist_ok=True)
            comp = hstack(row_imgs, gap=12)
            comp.save(tdir / "composite" / f"{sanitize(tid)}__overview_solutions.png")

def export_challenges_only(challenges: Dict[str, Dict[str, Any]], outdir: Path, scale: int, draw_grid: bool, make_composite: bool):
    """Export train pairs and test inputs only."""
    for tid, item in challenges.items():
        tdir = outdir / sanitize(tid)
        ensure_dir(tdir)
        rows = []
        #train pairs
        for i, pair in enumerate(item.get("train", []), start=1):
            input_grid = pair["input"]
            output_grid = pair["output"]
            im_in  = banner(grid_to_rgb(input_grid, scale, draw_grid),  f"{tid} • train#{i} • input")
            im_out = banner(grid_to_rgb(output_grid, scale, draw_grid), f"{tid} • train#{i} • output")
            pair_img = hstack([im_in, im_out], gap=12)
            pair_img.save(tdir / f"{sanitize(tid)}__train{i}_pair.png")
            rows.append(pair_img)
        #test inputs
        tests = item.get("test", [])
        #could be list of dicts with "input" or raw grids
        for j, t in enumerate(tests, start=1):
            grid = t["input"] if isinstance(t, dict) and "input" in t else t
            im_test = banner(grid_to_rgb(grid, scale, draw_grid), f"{tid} • test#{j} • input")
            im_test.save(tdir / f"{sanitize(tid)}__test{j}_input.png")
            rows.append(im_test)
        if make_composite and rows:
            (tdir / "composite").mkdir(exist_ok=True)
            comp = vstack(rows, gap=14)
            comp.save(tdir / "composite" / f"{sanitize(tid)}__overview_challenges.png")

def export_challenges_with_solutions(challenges: Dict[str, Dict[str, Any]], solutions: Dict[str, List[List[List[int]]]], outdir: Path, scale: int, draw_grid: bool, make_composite: bool):
    """Export train pairs, test inputs, and test ground-truth outputs if available."""
    for tid, item in challenges.items():
        tdir = outdir / sanitize(tid)
        ensure_dir(tdir)
        rows = []
        #train pairs
        for i, pair in enumerate(item.get("train", []), start=1):
            im_in  = banner(grid_to_rgb(pair["input"], scale, draw_grid),  f"{tid} • train#{i} • input")
            im_out = banner(grid_to_rgb(pair["output"], scale, draw_grid), f"{tid} • train#{i} • output")
            row_img = hstack([im_in, im_out], gap=12)
            row_img.save(tdir / f"{sanitize(tid)}__train{i}_pair.png")
            rows.append(row_img)
        #tests with GT if present
        test_list = item.get("test", [])
        gt_list = solutions.get(tid, [])
        for j, t in enumerate(test_list, start=1):
            test_grid = t["input"] if isinstance(t, dict) and "input" in t else t
            im_t = banner(grid_to_rgb(test_grid, scale, draw_grid), f"{tid} • test#{j} • input")
            if j-1 < len(gt_list):
                gt_grid = gt_list[j-1]
                im_gt = banner(grid_to_rgb(gt_grid, scale, draw_grid), f"{tid} • test#{j} • ground-truth")
                pair = hstack([im_t, im_gt], gap=12)
                pair.save(tdir / f"{sanitize(tid)}__test{j}_with_gt.png")
                rows.append(pair)
            else:
                im_t.save(tdir / f"{sanitize(tid)}__test{j}_input.png")
                rows.append(im_t)
        if make_composite and rows:
            (tdir / "composite").mkdir(exist_ok=True)
            comp = vstack(rows, gap=14)
            comp.save(tdir / "composite" / f"{sanitize(tid)}__overview_full.png")

def main():
    p = argparse.ArgumentParser(description="Export ARC-AGI JSON to PNGs")
    p.add_argument("--challenges", type=str, default=None, help="Path to *-challenges.json")
    p.add_argument("--solutions", type=str, default=None, help="Path to *-solutions.json")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for PNGs")
    p.add_argument("--scale", type=int, default=20, help="Cell scale (pixels per grid cell)")
    p.add_argument("--gridlines", action="store_true", help="Overlay gridlines")
    p.add_argument("--no-composite", action="store_true", help="Do not create overview collages per task")
    p.add_argument("--separate-only", action="store_true", help="Save separate PNGs only (same as --no-composite)")
    args = p.parse_args()

    if not args.challenges and not args.solutions:
        raise SystemExit("Provide at least one of --challenges or --solutions")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw_grid = args.gridlines
    make_comp = not (args.no_composite or args.separate_only)

    challenges = None
    solutions = None

    if args.challenges:
        with open(args.challenges, "r") as f:
            challenges = json.load(f)
        #quick format check
        if detect_format(challenges) != "challenges":
            raise SystemExit("File passed to --challenges does not look like a challenges JSON")

    if args.solutions:
        with open(args.solutions, "r") as f:
            solutions = json.load(f)
        if detect_format(solutions) != "solutions":
            raise SystemExit("File passed to --solutions does not look like a solutions JSON")

    if challenges and solutions:
        export_challenges_with_solutions(challenges, solutions, outdir, args.scale, draw_grid, make_comp)
    elif challenges:
        export_challenges_only(challenges, outdir, args.scale, draw_grid, make_comp)
    elif solutions:
        export_solutions_only(solutions, outdir, args.scale, draw_grid, make_comp)

    print(f"Done. PNGs written to: {outdir}")

if __name__ == "__main__":
    main()