# Video Panorama Stitcher

A simple Python project: run the script, enter a video path, and it extracts frames and stitches them into a panorama.

## Installation

```bash
cd /root/projects/video-panorama-stitcher
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Interactive mode:

```bash
. .venv/bin/activate
python panorama_from_video.py
```

Then enter the path to your video in the console, for example:

```text
/root/videos/test.mp4
```

Or pass the path as a command-line argument:

```bash
python panorama_from_video.py /root/videos/test.mp4
```

## Output

- Extracted frames are saved to `output/<video-name>/frames/`
- The panorama is saved to `output/<video-name>/panorama.jpg`

## Settings

```bash
python panorama_from_video.py video.mp4 --frame-step 15 --max-frames 40 --resize-width 960
```

- `--method strip` — fast mode (default): takes the central vertical strip from each frame and combines them into a wide panorama.
- `--method stitch` — automatically stitches frames using matching features with OpenCV. It can produce better results, but may be slow and does not work with every video.
- `--frame-step 15` — extract every 15th frame.
- `--max-frames 40` — maximum number of frames to use for stitching.
- `--resize-width 960` — resize each frame to a width of 960 pixels before stitching.

If the video was recorded without smooth camera movement or rotation, or if the frames have too little overlap, OpenCV's stitching algorithm may fail to create a panorama. In that case, use the fast `--method strip` mode, decrease `--frame-step`, or increase `--max-frames`.