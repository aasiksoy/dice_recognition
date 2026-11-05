import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dice_detector import cluster_pips, create_blob_detector, detect_dice, parse_source, resize_to_width

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
