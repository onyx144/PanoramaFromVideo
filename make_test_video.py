#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np

out_dir = Path('test_assets')
out_dir.mkdir(exist_ok=True)
video_path = out_dir / 'synthetic_panorama_scan.mp4'

canvas_h, canvas_w = 420, 1500
view_w, view_h = 520, 360
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# gradient sky/background
for x in range(canvas_w):
    color = int(60 + 120 * x / canvas_w)
    canvas[:, x] = (color, 120, 220 - color // 2)

# ground
cv2.rectangle(canvas, (0, 300), (canvas_w, canvas_h), (70, 160, 80), -1)

# distinctive objects across the canvas
for i, x in enumerate(range(80, canvas_w - 80, 140)):
    y = 245 - (i % 3) * 20
    cv2.circle(canvas, (x, y), 35 + (i % 2) * 8, (30 + i * 20 % 200, 40 + i * 30 % 180, 220 - i * 15 % 180), -1)
    cv2.putText(canvas, str(i + 1), (x - 14, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

for x in range(0, canvas_w, 100):
    cv2.line(canvas, (x, 0), (x, canvas_h), (255, 255, 255), 1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(video_path), fourcc, 12, (view_w, view_h))
for x in np.linspace(0, canvas_w - view_w, 72).astype(int):
    frame = canvas[20:20 + view_h, x:x + view_w].copy()
    writer.write(frame)
writer.release()
print(video_path)
