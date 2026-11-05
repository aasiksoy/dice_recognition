import argparse
import sys

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

WINDOW_NAME = "Dice Detector"
DEFAULT_WIDTH = 800
DEFAULT_EPS = 60.0


def create_blob_detector(min_area=30, max_area=2000, min_circularity=0.6, min_inertia=0.3):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia
    params.filterByColor = True
    params.blobColor = 0
    return cv2.SimpleBlobDetector_create(params)


def put_label(img, text, pos, color=(0, 0, 255), scale=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 4, cv2.LINE_AA)


def cluster_pips(points, eps=DEFAULT_EPS):
    if len(points) == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=1).fit(points).labels_


def detect_dice(frame, detector, eps=DEFAULT_EPS):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    keypoints = detector.detect(gray)
    if not keypoints:
        return [], keypoints

    pts = np.array([kp.pt for kp in keypoints])
    labels = cluster_pips(pts, eps)

    dice = []
    for die_id in np.unique(labels):
        cluster = pts[labels == die_id]
        x_min, y_min = cluster.min(axis=0).astype(int)
        x_max, y_max = cluster.max(axis=0).astype(int)
        cx, cy = cluster.mean(axis=0).astype(int)
        dice.append({
            "value": len(cluster),
            "center": (int(cx), int(cy)),
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
        })

    dice.sort(key=lambda d: d["center"])
    return dice, keypoints


def annotate(frame, dice, keypoints, padding=20):
    for die in dice:
        x_min, y_min, x_max, y_max = die["bbox"]
        cv2.rectangle(frame, (x_min - padding, y_min - padding),
                      (x_max + padding, y_max + padding), (0, 255, 0), 3)
        cx, cy = die["center"]
        put_label(frame, str(die["value"]), (cx - 20, cy))

    frame = cv2.drawKeypoints(frame, keypoints, None, (255, 0, 0),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if dice:
        total = sum(d["value"] for d in dice)
        put_label(frame, f"Total: {total}", (50, 80), color=(50, 200, 50))
    return frame


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def parse_source(source):
    return int(source) if source.isdigit() else source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect dice and count pips in a video, camera stream, or image.")
    parser.add_argument("source", nargs="?", default="0",
                        help="video file, image file, or camera index (default: 0)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"resize frames to this width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help=f"max pip distance within one die, in pixels (default: {DEFAULT_EPS})")
    parser.add_argument("--min-area", type=float, default=30, help="minimum pip area (default: 30)")
    parser.add_argument("--max-area", type=float, default=2000, help="maximum pip area (default: 2000)")
    parser.add_argument("--output", help="save the annotated result to this file")
    parser.add_argument("--no-display", action="store_true", help="do not open a preview window")
    return parser.parse_args(argv)


def process_image(path, detector, args):
    frame = cv2.imread(path)
    if frame is None:
        return False
    frame = resize_to_width(frame, args.width)
    dice, keypoints = detect_dice(frame, detector, args.eps)
    result = annotate(frame, dice, keypoints)
    print(f"Dice: {[d['value'] for d in dice]}  Total: {sum(d['value'] for d in dice)}")

    if args.output:
        cv2.imwrite(args.output, result)
    if not args.no_display:
        cv2.imshow(WINDOW_NAME, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return True


def process_video(source, detector, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{source}'", file=sys.stderr)
        return 1

    writer = None
    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_to_width(frame, args.width)
            dice, keypoints = detect_dice(frame, detector, args.eps)
            result = annotate(frame, dice, keypoints)

            if args.output:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    h, w = result.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(result)

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, result)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args = parse_args(argv)
    detector = create_blob_detector(min_area=args.min_area, max_area=args.max_area)
    source = parse_source(args.source)

    if isinstance(source, str) and process_image(source, detector, args):
        return 0
    return process_video(source, detector, args)


if __name__ == "__main__":
    sys.exit(main())
