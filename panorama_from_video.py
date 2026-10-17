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
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-._")
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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        f"Не удалось открыть видео: {video_path}. "
        f"Поддерживаемые контейнеры: {SUPPORTED_VIDEO_EXTENSIONS}. "
        "Если это MOV с редким кодеком, перекодируй его в H.264/AAC или ProRes/HEVC, доступный FFmpeg/OpenCV."
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
        raise RuntimeError("Нужно минимум 2 кадра для панорамы. Попробуй меньший --frame-step или другое видео.")

    for saved_index, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"frame_{saved_index:04d}.jpg"), frame)
    return frames


def stitch_frames(frames: list[np.ndarray], mode: str = "panorama") -> tuple[int, np.ndarray | None]:
    stitcher_mode = cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher_create(stitcher_mode)
    status, pano = stitcher.stitch(frames)
    if status == cv2.Stitcher_OK and pano is not None:
        return status, crop_black_borders(pano)
    return status, None


def fallback_strip_panorama(frames: list[np.ndarray], *, slice_width_ratio: float = 0.22) -> np.ndarray:
    """Puts the full first frame on the left; later frames contribute a center strip."""
    min_h = min(f.shape[0] for f in frames)
    pieces = []
    for i, frame in enumerate(frames):
        if frame.shape[0] != min_h:
            ratio = min_h / frame.shape[0]
            frame = cv2.resize(frame, (max(1, int(frame.shape[1] * ratio)), min_h), interpolation=cv2.INTER_AREA)
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
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    out_dir = output_root / safe_name(str(video_path))
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Видео: {video_path}")
    print(f"Кадры: {frames_dir}")
    frames = extract_frames(
        video_path,
        frames_dir,
        frame_step=max(1, frame_step),
        max_frames=max(2, max_frames),
        resize_width=resize_width,
    )
    print(f"Извлечено кадров: {len(frames)}")

    if method == "strip":
        print("Создаю strip-панораму: полный первый кадр и полосы остальных...")
        pano = fallback_strip_panorama(frames)
    else:
        status, pano = stitch_frames(frames, mode="panorama")
        if pano is None:
            print(f"OpenCV PANORAMA не склеил: {STATUS_NAMES.get(status, status)}")
            status, pano = stitch_frames(frames, mode="scans")

        if pano is None:
            print(f"OpenCV SCANS не склеил: {STATUS_NAMES.get(status, status)}")
            if not fallback:
                raise RuntimeError("Панорама не создана. Попробуй --frame-step меньше, --max-frames больше или включи fallback.")
            print("Создаю fallback strip-панораму из центральных полос кадров...")
            pano = fallback_strip_panorama(frames)

    out_path = out_dir / "panorama.jpg"
    cv2.imwrite(str(out_path), pano, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Verify readable image.
    with Image.open(out_path) as im:
        print(f"Панорама: {out_path}")
        print(f"Размер: {im.size[0]}x{im.size[1]}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut video into frames and create panorama")
    parser.add_argument("video", nargs="?", help="Путь к видео. Если не указан, скрипт спросит в консоли.")
    parser.add_argument("--frame-step", type=int, default=55, help="Брать каждый N-й кадр. По умолчанию 15.")
    parser.add_argument("--max-frames", type=int, default=60, help="Максимум кадров. По умолчанию 40.")
    parser.add_argument("--resize-width", type=int, default=960, help="Ширина кадра перед склейкой. 0 = не менять.")
    parser.add_argument("--output", default="output", help="Папка результата. По умолчанию output.")
    parser.add_argument("--method", choices=["strip", "stitch"], default="strip", help="strip = быстро и стабильно, stitch = OpenCV авто-склейка, но может быть медленно. По умолчанию strip.")
    parser.add_argument("--no-fallback", action="store_true", help="Не делать strip-панораму, если OpenCV не склеил.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video or input("Введите путь/название видео: ").strip().strip('"').strip("'")
    if not video:
        print("Видео не указано", file=sys.stderr)
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
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
