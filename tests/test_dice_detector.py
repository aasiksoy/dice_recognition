from pathlib import Path

import cv2
import numpy as np
import pytest

from dice_detector import (
    annotate,
    cluster_pips,
    create_blob_detector,
    detect_dice,
    main,
    parse_args,
    parse_source,
    resize_to_width,
)

PIP_LAYOUTS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
}


def draw_die(img, center, value, size=80, pip_radius=7):
    cx, cy = center
    half = size // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), -1)
    step = size // 4
    for dx, dy in PIP_LAYOUTS[value]:
        cv2.circle(img, (cx + dx * step, cy + dy * step), pip_radius, (0, 0, 0), -1)


def make_scene(values):
    img = np.full((300, 200 * len(values), 3), 90, dtype=np.uint8)
    for i, value in enumerate(values):
        draw_die(img, (100 + 200 * i, 150), value)
    return img


@pytest.mark.parametrize("value", range(1, 7))
def test_single_die_value(value):
    dice, _ = detect_dice(make_scene([value]), create_blob_detector())
    assert [d["value"] for d in dice] == [value]


def test_multiple_dice_sorted_left_to_right():
    values = [3, 6, 1, 5]
    dice, _ = detect_dice(make_scene(values), create_blob_detector())
    assert [d["value"] for d in dice] == values


def test_empty_frame_returns_no_dice():
    frame = np.full((200, 200, 3), 90, dtype=np.uint8)
    dice, keypoints = detect_dice(frame, create_blob_detector())
    assert dice == []
    assert len(keypoints) == 0


def test_cluster_pips_groups_by_distance():
    points = np.array([[0, 0], [10, 0], [500, 500], [510, 500]])
    labels = cluster_pips(points, eps=50)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_pips_empty():
    assert len(cluster_pips(np.empty((0, 2)))) == 0


def test_resize_to_width_keeps_aspect_ratio():
    frame = np.zeros((400, 1600, 3), dtype=np.uint8)
    assert resize_to_width(frame, 800).shape[:2] == (200, 800)


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("dice.mp4") == "dice.mp4"


@pytest.fixture
def detector():
    return create_blob_detector()


@pytest.fixture
def scene_image(tmp_path):
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), make_scene([2, 4, 6]))
    return path


def test_die_center_and_bbox(detector):
    dice, _ = detect_dice(make_scene([5]), detector)
    (die,) = dice
    cx, cy = die["center"]
    assert abs(cx - 100) <= 2 and abs(cy - 150) <= 2
    x_min, y_min, x_max, y_max = die["bbox"]
    assert x_min < cx < x_max and y_min < cy < y_max


def test_max_area_filters_out_large_pips():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(img, (150, 150), 40, (0, 0, 0), -1)
    dice, _ = detect_dice(img, create_blob_detector(max_area=2000))
    assert dice == []


def test_small_eps_splits_die(detector):
    dice, _ = detect_dice(make_scene([6]), detector, eps=5)
    assert len(dice) == 6
    assert all(d["value"] == 1 for d in dice)


def test_annotate_draws_on_frame(detector):
    frame = make_scene([3])
    dice, keypoints = detect_dice(frame.copy(), detector)
    result = annotate(frame.copy(), dice, keypoints)
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)


def test_annotate_without_dice_leaves_frame_unchanged():
    frame = np.full((200, 200, 3), 90, dtype=np.uint8)
    result = annotate(frame.copy(), [], [])
    assert np.array_equal(result, frame)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.source == "0"
    assert args.width == 800
    assert args.eps == 60.0
    assert args.output is None
    assert args.no_display is False


def test_parse_args_custom():
    args = parse_args(["clip.mp4", "--width", "640", "--eps", "45", "--output", "out.mp4", "--no-display"])
    assert (args.source, args.width, args.eps, args.output, args.no_display) == (
        "clip.mp4", 640, 45.0, "out.mp4", True)


def test_main_image_prints_result_and_saves_output(scene_image, tmp_path, capsys):
    output = tmp_path / "out.png"
    code = main([str(scene_image), "--no-display", "--width", "600", "--output", str(output)])
    assert code == 0
    assert "Dice: [2, 4, 6]  Total: 12" in capsys.readouterr().out
    assert cv2.imread(str(output)) is not None


def test_main_video_writes_output(tmp_path):
    video = tmp_path / "dice.avi"
    frame = make_scene([1, 3])
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for _ in range(5):
        writer.write(frame)
    writer.release()

    output = tmp_path / "out.mp4"
    code = main([str(video), "--no-display", "--width", "400", "--output", str(output)])
    assert code == 0
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def test_main_missing_source_returns_error(tmp_path, capsys):
    code = main([str(tmp_path / "missing.mp4"), "--no-display"])
    assert code == 1
    assert "could not open source" in capsys.readouterr().err


DATA_DIR = Path(__file__).parent / "data"

REAL_IMAGES = {
    "dice1.jpg": [3, 6],
    "dice2.jpeg": [6, 6],
    "dice3.jpeg": [2, 2],
    "dice4.jpeg": [2, 5],
}


@pytest.mark.parametrize("filename, expected", REAL_IMAGES.items())
def test_real_image(filename, expected, detector):
    frame = cv2.imread(str(DATA_DIR / filename))
    assert frame is not None, f"could not read {filename}"
    dice, _ = detect_dice(resize_to_width(frame, 800), detector)
    assert [d["value"] for d in dice] == expected


def test_shadow_is_not_counted_as_pip(detector):
    frame = resize_to_width(cv2.imread(str(DATA_DIR / "dice4.jpeg")), 800)
    loose, _ = detect_dice(frame, detector, min_contrast=0)
    strict, _ = detect_dice(frame, detector)
    assert sum(d["value"] for d in loose) > sum(d["value"] for d in strict) == 7
