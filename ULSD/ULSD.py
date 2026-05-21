import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from yacs.config import CfgNode


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from config.cfg import PATH_KEYS, resolve_path
from MaDCoW.src import CameraConfig
from MaDCoW.src.camera import Camera
from network.dataset import Dataset
from network.ulsd import ULSD
from test import save_lines
from util import bezier as bez


LINE_SAMPLE_POINTS = 128
DEFAULT_IMAGE = BASE_DIR / "dataset" / "my_image" / "room.jpg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "MDCoW" / "data"
DEFAULT_SENSOR_WIDTH_MM = 36.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ULSD on an image and export MaDCoW line annotations."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument("--model_name", default="pinhole.pkl", help="ULSD model filename.")
    parser.add_argument("--order", type=int, default=4, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id. Use -1 for CPU.")
    parser.add_argument("--config_path", default="config", help="ULSD config folder.")
    parser.add_argument("--config_file", default="default.yaml", help="ULSD config filename.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--marked_image", default="", help="Marked image output path.")
    parser.add_argument("--json", default="", help="MaDCoW JSON output path.")

    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--junc_score_thresh", type=float, default=None)
    parser.add_argument("--line_score_thresh", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top K scored lines after thresholding.")

    parser.add_argument("--focal_length_mm", type=float, default=13.0)
    parser.add_argument(
        "--sensor_width_mm",
        type=float,
        default=DEFAULT_SENSOR_WIDTH_MM,
        help="Sensor width used to convert focal length to horizontal FOV.",
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=None,
        help="Override FOV directly. If omitted, FOV is computed from focal length and sensor width.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def horizontal_fov_deg(focal_length_mm, sensor_width_mm):
    if focal_length_mm <= 0:
        raise ValueError(f"focal_length_mm must be positive; got {focal_length_mm}.")
    if sensor_width_mm <= 0:
        raise ValueError(f"sensor_width_mm must be positive; got {sensor_width_mm}.")
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_length_mm)))


def load_cfg(args):
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    yaml_file = config_path / args.config_file

    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = CfgNode.load_cfg(f)

    cfg.defrost()
    cfg.dataset_name = str(Path(args.image).resolve())
    cfg.order = args.order
    cfg.gpu = args.gpu
    cfg.model_name = args.model_name
    cfg.version = ".".join(cfg.model_name.split(".")[:-1])
    cfg.config_path = str(config_path)
    cfg.config_file = args.config_file
    cfg.test_dataset_path = str(Path(args.image).resolve())
    cfg.groundtruth_path = str(Path(args.image).resolve().parent)
    cfg.output_path = str(Path(args.output_dir).resolve())
    cfg.figure_path = str(Path(args.output_dir).resolve())
    cfg.log_path = os.path.join(cfg.log_path, cfg.version)
    cfg.image_size = tuple(cfg.image_size)
    cfg.heatmap_size = tuple(cfg.heatmap_size)
    if args.score_thresh is not None:
        cfg.score_thresh = float(args.score_thresh)
    if args.junc_score_thresh is not None:
        cfg.junc_score_thresh = float(args.junc_score_thresh)
    if args.line_score_thresh is not None:
        cfg.line_score_thresh = float(args.line_score_thresh)

    for key in PATH_KEYS:
        cfg[key] = resolve_path(cfg[key])
    cfg.freeze()
    return cfg


def rescale_lines_to_image(lines, image_shape, cfg):
    lines = lines.astype(np.float64, copy=True)
    height, width = image_shape[:2]
    sx = width / cfg.heatmap_size[0]
    sy = height / cfg.heatmap_size[1]
    lines[:, :, 0] *= sx
    lines[:, :, 1] *= sy
    return lines


def choose_lines(line_pred, line_score, score_thresh, top_k):
    keep = line_score > score_thresh
    lines = line_pred[keep]
    scores = line_score[keep]
    if top_k and top_k > 0 and len(scores) > top_k:
        order = np.argsort(scores)[::-1][:top_k]
        lines = lines[order]
        scores = scores[order]
    return lines, scores


def lines_to_madcow_annotations(lines_px, camera):
    annotations = []
    if len(lines_px) == 0:
        return annotations

    samples_list = bez.interp_line(lines_px, num=LINE_SAMPLE_POINTS)
    for samples in samples_list:
        x = np.clip(samples[:, 0], 0.0, camera.cfg.width - 1.0)
        y = np.clip(samples[:, 1], 0.0, camera.cfg.height - 1.0)
        lam, phi = camera.pixel_to_direction(x, y)
        points_dir = [
            [float(lam_i), float(phi_i)]
            for lam_i, phi_i in zip(lam, phi)
        ]
        annotations.append({"points_dir": points_dir})
    return annotations


def relative_path(path, base_dir):
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def main():
    args = parse_args()
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marked_image_path = Path(args.marked_image).resolve() if args.marked_image else output_dir / f"{image_path.stem}_ulsd_lines.png"
    json_path = Path(args.json).resolve() if args.json else output_dir / f"{image_path.stem}.json"
    marked_image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fov_deg = args.fov_deg
    if fov_deg is None:
        fov_deg = horizontal_fov_deg(args.focal_length_mm, args.sensor_width_mm)
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov_deg must be in (0, 180); got {fov_deg}.")

    cfg = load_cfg(args)
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.gpu}" if use_gpu else "cpu")

    model = ULSD(cfg).to(device)
    model_filename = Path(cfg.model_path) / cfg.model_name
    checkpoint = torch.load(model_filename, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
    model.load_state_dict(state_dict)
    model.lpn.junc_score_thresh = float(cfg.junc_score_thresh)
    model.lpn.line_score_thresh = float(cfg.line_score_thresh)
    model.eval()

    dataset = Dataset(str(image_path), cfg, with_label=False)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No supported image found at {image_path}")
    loader = Data.DataLoader(dataset=dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    with torch.no_grad():
        images = next(iter(loader)).to(device)
        _, _, line_preds, line_scores = model(images)
        line_pred = line_preds[0].detach().cpu().numpy()
        line_score = line_scores[0].detach().cpu().numpy()

    selected_lines, selected_scores = choose_lines(
        line_pred=line_pred,
        line_score=line_score,
        score_thresh=float(cfg.score_thresh),
        top_k=args.top_k,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    selected_lines_px = rescale_lines_to_image(selected_lines, image.shape, cfg)
    save_lines(image.copy(), selected_lines.copy(), str(marked_image_path), cfg, fast=True)

    height, width = image.shape[:2]
    camera = Camera(CameraConfig(fov_deg=float(fov_deg), width=width, height=height))
    madcow_lines = lines_to_madcow_annotations(selected_lines_px, camera)

    payload = {
        "image_path": relative_path(image_path, json_path.parent),
        "fov_deg": float(fov_deg),
        "focal_length_mm": float(args.focal_length_mm),
        "sensor_width_mm": float(args.sensor_width_mm),
        "lines": madcow_lines,
        "regions": [],
    }
    json_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    print(f"use_gpu: {use_gpu}")
    print(f"image: {image_path}")
    print(f"fov_deg: {fov_deg:.6f}")
    print(f"candidates: {len(line_pred)}")
    print(f"selected_lines: {len(selected_lines)}")
    print(f"marked_image: {marked_image_path}")
    print(f"madcow_json: {json_path}")


if __name__ == "__main__":
    main()
