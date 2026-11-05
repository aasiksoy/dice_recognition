# Dice Detector – OpenCV + DBSCAN

[![tests](https://github.com/aasiksoy/dice_recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/aasiksoy/dice_recognition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time dice recognition project that detects pips (dots) on dice using
**OpenCV SimpleBlobDetector** and clusters them into individual dice using
**DBSCAN**. It counts the pips on each die, draws a box around it, and shows
each die's value and the total.

## Features

- Detects pips with blob detection
- Groups pips into dice with **DBSCAN clustering**
- Draws bounding boxes and labels each die with its value
- Displays the total of all dice
- Works with video files, images, and live camera streams
- Can save the annotated output to a file
- Detection parameters adjustable from the command line

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/aasiksoy/dice_recognition.git
cd dice_recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python dice_detector.py
python dice_detector.py dice.mp4
python dice_detector.py photo.jpg --output result.jpg
python dice_detector.py dice.mp4 --output annotated.mp4 --no-display
```

| Command | What it does |
| --- | --- |
| `python dice_detector.py` | Uses the default webcam (camera `0`) |
| `python dice_detector.py dice.mp4` | Processes a video file |
| `python dice_detector.py photo.jpg --output result.jpg` | Processes a single image and saves the result |
| `python dice_detector.py dice.mp4 --output annotated.mp4 --no-display` | Saves an annotated video without opening a window |

Press **q** to quit the video window.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `source` | `0` | Video file, image file, or camera index |
| `--width` | `800` | Frames are resized to this width before detection |
| `--eps` | `60` | Maximum distance (pixels) between pips of the same die |
| `--min-area` | `30` | Minimum pip area in pixels |
| `--max-area` | `2000` | Maximum pip area in pixels |
| `--min-contrast` | `80` | How much darker a pip must be than the area around it |
| `--output` | – | Save the annotated image or video to this path |
| `--no-display` | off | Run without opening a preview window |

## How it works

1. **Preprocessing** – each frame is resized, converted to grayscale and lightly blurred.
2. **Pip detection** – `cv2.SimpleBlobDetector` finds dark, round blobs filtered by area, circularity and inertia.
   Blobs that are not clearly darker than their surroundings, such as shadows next to a die, are discarded.
3. **Clustering** – pip centers are grouped with `DBSCAN(min_samples=1)`. Pips closer than `eps` pixels belong to the same die.
4. **Counting** – the number of pips in each cluster is that die's value, and the values are summed for the total.

## Tuning tips

- **Dice are merged together:** lower `--eps` or space the dice further apart.
- **One die is split into several:** raise `--eps`, especially for large dice or high resolutions.
- **Pips are missed:** lower `--min-area`, or improve lighting and contrast.
- **Shadows or dark spots counted as pips:** raise `--min-contrast`.
- **Faint pips ignored:** lower `--min-contrast`.
- **False detections:** raise `--min-area` or use a plain, light background.

The detector works best with light dice that have dark pips, photographed from above under even lighting.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover both synthetic dice drawn with OpenCV and real photos in
`tests/data/`. To add a real photo, place it in that folder and add its
expected values (left to right) to `REAL_IMAGES` in the test file.

## Project structure

```
dice_recognition/
├── dice_detector.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── tests/
│   ├── data/
│   └── test_dice_detector.py
└── .github/workflows/tests.yml
```

## License

Released under the [MIT License](LICENSE).
