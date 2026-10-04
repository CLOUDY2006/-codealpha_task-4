import argparse
import os
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

WINDOW = "Object Detection and Tracking"
FONT = cv2.FONT_HERSHEY_SIMPLEX
TRAIL_LENGTH = 30

PALETTE = [
    (66, 135, 245), (60, 180, 75), (230, 25, 75), (255, 165, 0), (145, 30, 180),
    (70, 200, 200), (240, 50, 230), (128, 128, 0), (0, 130, 200), (210, 140, 90),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument("--source", default="0", help="0 for the webcam, or the path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (downloaded automatically)")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence")
    parser.add_argument("--classes", nargs="*", default=None, help="only track these classes, e.g. person car")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--save", action="store_true", help="save the result as output.mp4")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    return parser.parse_args()


def color_for(track_id):
    return PALETTE[track_id % len(PALETTE)]


def draw_label(frame, text, x, y, color):
    (width, height), baseline = cv2.getTextSize(text, FONT, 0.55, 1)
    top = max(y - height - baseline - 6, 0)
    cv2.rectangle(frame, (x, top), (x + width + 8, top + height + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + height + 2), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(frame, fps, current, seen):
    lines = [f"FPS: {fps:.1f}", f"In frame: {current}"]
    ranked = sorted(seen.items(), key=lambda item: -len(item[1]))[:5]
    lines += [f"{name}: {len(ids)} seen" for name, ids in ranked]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (240, 20 + 24 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(frame, line, (20, 34 + 24 * index), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def open_source(source):
    if source.isdigit():
        index = int(source)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        return cv2.VideoCapture(index, backend), True

    return cv2.VideoCapture(source), False


def main():
    args = parse_args()
    model = YOLO(args.model)

    class_ids = None
    if args.classes:
        names = {name.lower(): idx for idx, name in model.names.items()}
        unknown = [c for c in args.classes if c.lower() not in names]
        if unknown:
            print("Unknown class names:", ", ".join(unknown))
            print("Available classes:", ", ".join(sorted(names)))
            return
        class_ids = [names[c.lower()] for c in args.classes]

    cap, is_webcam = open_source(args.source)
    if not cap.isOpened():
        print("Could not open the video source:", args.source)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    seen = defaultdict(set)
    fps = 0.0
    last_time = time.time()

    print("Running. Press q to quit, space to pause.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if is_webcam:
            frame = cv2.flip(frame, 1)

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            classes=class_ids,
            tracker=args.tracker,
            verbose=False,
        )

        boxes = results[0].boxes
        current = 0

        if boxes is not None and boxes.id is not None:
            coords = boxes.xyxy.int().cpu().tolist()
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for (x1, y1, x2, y2), track_id, class_id, conf in zip(coords, ids, classes, confidences):
                name = model.names[class_id]
                color = color_for(track_id)
                current += 1
                seen[name].add(track_id)

                trails[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                points = list(trails[track_id])
                for start, end in zip(points, points[1:]):
                    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"{name} #{track_id} {conf:.2f}", x1, y1, color)

        now = time.time()
        elapsed = max(now - last_time, 1e-6)
        fps = 1 / elapsed if fps == 0 else 0.9 * fps + 0.1 / elapsed
        last_time = now

        draw_panel(frame, fps, current, seen)

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == 32:
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (32, ord("q")):
                        break
                if key == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
        print("Saved output.mp4")
    cv2.destroyAllWindows()

    print("Unique objects seen:")
    for name, ids in sorted(seen.items(), key=lambda item: -len(item[1])):
        print(f"  {name}: {len(ids)}")


if __name__ == "__main__":
    main()