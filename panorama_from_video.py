```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


STATUS_NAMES = {
    0: "OK",
    1: "ERR_NEED_MORE_IMGS",
    2: "ERR_HOMOGRAPHY_EST_FAIL",
    3: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}

SUPPORTED_VIDEO_EXTENSIONS = ".mp4, .mov, .m4v, .avi, .mkv, .webm"


def safe_name(value: str) -> str:
    name = Path(value).stem or "video"
    name = re.sub(r"[^a-zA-Z0-9\_.-]+", "-", name).strip("-._")
    return name or "video"


def resize_keep_aspect(frame: np.ndarray, width: int | None) -> np.ndarray:
    if not width or frame.shape[1] <= width:
        return frame
    ratio = width / frame.shape[1]
    height = max(1, int(frame.shape[0] * ratio))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        return image
    return image[y:y + h, x:x + w]


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """Open MP4/MOV/etc. through FFmpeg when available, then fall back to OpenCV defaults."""
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(str(video_path), backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError(
        f"Failed to open video: {video_path}. "
        f"Supported containers: {SUPPORTED_VIDEO_EXTENSIONS}. "
        "If this is a MOV file with an uncommon codec, convert it to "
        "H.264/AAC or ProRes/HEVC supported by FFmpeg/OpenCV."
    )


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    frame_step: int,
    max_frames: int,
    resize_width: int | None,
) -> list[np.ndarray]:
    cap = open_video_capture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index == 0 or frame_index % frame_step == 0:
            frame = resize_keep_aspect(frame, resize_width)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_index += 1

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            "At least 2 frames are required to create a panorama. "
            "Try a smaller --frame-step value or use a different video."
        )

    for saved_index, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"frame_{saved_index:04d}.jpg"), frame)
    return frames


def stitch_frames(
    frames: list[np.ndarray], mode: str = "panorama"
) -> tuple[int, np.ndarray | None]:
    stitcher_mode = (
        cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS
    )
    stitcher = cv2.Stitcher_create(stitcher_mode)
    status, pano = stitcher.stitch(frames)
    if status == cv2.Stitcher_OK and pano is not None:
        return status, crop_black_borders(pano)
    return status, None


def fallback_strip_panorama(
    frames: list[np.ndarray], *, slice_width_ratio: float = 0.22
) -> np.ndarray:
    """Keep the entire first frame on the left; use a center strip from each subsequent frame."""
    min_h = min(f.shape[0] for f in frames)
    pieces = []
    for i, frame in enumerate(frames):
        if frame.shape[0] != min_h:
            ratio = min_h / frame.shape[0]
            frame = cv2.resize(
                frame,
                (max(1, int(frame.shape[1] * ratio)), min_h),
                interpolation=cv2.INTER_AREA,
            )
        if i == 0:
            pieces.append(frame)
            continue
        w = frame.shape[1]
        slice_w = max(8, int(w * slice_width_ratio))
        x1 = max(0, (w - slice_w) // 2)
        pieces.append(frame[:, x1:x1 + slice_w])
    return cv2.hconcat(pieces)


def make_panorama(
    video_path: Path,
    *,
    output_root: Path,
    frame_step: int,
    max_frames: int,
    resize_width: int | None,
    fallback: bool,
    method: str,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = output_root / safe_name(str(video_path))
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Frames directory: {frames_dir}")
    frames = extract_frames(
        video_path,
        frames_dir,
        frame_step=max(1, frame_step),
        max_frames=max(2, max_frames),
        resize_width=resize_width,
    )
    print(f"Extracted frames: {len(frames)}")

    if method == "strip":
        print(
            "Creating a strip panorama: full first frame followed by strips "
            "from the remaining frames..."
        )
        pano = fallback_strip_panorama(frames)
    else:
        status, pano = stitch_frames(frames, mode="panorama")
        if pano is None:
            print(
                f"OpenCV PANORAMA stitching failed: "
                f"{STATUS_NAMES.get(status, status)}"
            )
            status, pano = stitch_frames(frames, mode="scans")

        if pano is None:
            print(
                f"OpenCV SCANS stitching failed: "
                f"{STATUS_NAMES.get(status, status)}"
            )
            if not fallback:
                raise RuntimeError(
                    "Failed to create a panorama. Try a smaller --frame-step "
                    "value, a larger --max-frames value, or enable the fallback."
                )
            print(
                "Creating a fallback strip panorama using center strips "
                "from the frames..."
            )
            pano = fallback_strip_panorama(frames)

    out_path = out_dir / "panorama.jpg"
    cv2.imwrite(str(out_path), pano, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Verify that the output image is readable.
    with Image.open(out_path) as im:
        print(f"Panorama: {out_path}")
        print(f"Dimensions: {im.size[0]}x{im.size[1]}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and create a panorama"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to the video. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=55,
        help="Extract every Nth frame. Default: 55.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Maximum number of frames to extract. Default: 60.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Frame width before stitching. 0 = keep original dimensions.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory. Default: output.",
    )
    parser.add_argument(
        "--method",
        choices=["strip", "stitch"],
        default="strip",
        help=(
            "strip = fast and reliable; stitch = automatic OpenCV stitching, "
            "which may be slow. Default: strip."
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not create a strip panorama if OpenCV stitching fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = (
        args.video
        or input("Enter the video path or filename: ").strip().strip('"').strip("'")
    )
    if not video:
        print("No video specified.", file=sys.stderr)
        return 2
    try:
        make_panorama(
            Path(video).expanduser().resolve(),
            output_root=Path(args.output).expanduser().resolve(),
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            resize_width=args.resize_width if args.resize_width > 0 else None,
            fallback=not args.no_fallback,
            method=args.method,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```