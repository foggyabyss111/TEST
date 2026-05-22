import argparse
import json
import os
import shutil

import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return fps if fps > 0 else 16.0, frames


def write_video(video_path, fps, frames):
    if not frames:
        raise ValueError("Cannot write empty frame list.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {video_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def prepare_segments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    fps, frames = read_video(args.input_video)
    total_frames = len(frames)
    segment_frames = args.segment_frames

    if segment_frames % 4 != 1:
        raise ValueError("segment_frames must satisfy 4n+1.")
    if total_frames < segment_frames:
        raise ValueError(
            f"Video has {total_frames} frames, smaller than segment_frames={segment_frames}."
        )

    seg1_start = 0
    seg2_start = total_frames - segment_frames
    overlap = segment_frames - seg2_start
    mid_skip = overlap // 2
    seg1_keep = seg2_start + mid_skip

    seg1_frames = frames[seg1_start:seg1_start + segment_frames]
    seg2_frames = frames[seg2_start:seg2_start + segment_frames]

    seg1_video = os.path.join(args.output_dir, "segment1.mp4")
    seg2_video = os.path.join(args.output_dir, "segment2.mp4")
    write_video(seg1_video, fps, seg1_frames)
    write_video(seg2_video, fps, seg2_frames)

    seg1_origin = os.path.join(args.output_dir, "segment1_origin.jpg")
    seg2_origin = os.path.join(args.output_dir, "segment2_origin.jpg")
    cv2.imwrite(seg1_origin, seg1_frames[0])
    cv2.imwrite(seg2_origin, seg2_frames[0])

    shutil.copy2(args.edited_image, os.path.join(args.output_dir, "segment1_edit.jpg"))
    shutil.copy2(args.edited_image, os.path.join(args.output_dir, "segment2_edit.jpg"))

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "segment_frames": segment_frames,
        "segment1_start": seg1_start,
        "segment2_start": seg2_start,
        "overlap": overlap,
        "segment1_keep": seg1_keep,
        "segment2_skip": mid_skip,
    }
    with open(os.path.join(args.output_dir, "segments_meta.json"), "w", encoding="ascii") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


def stitch_segments(args):
    with open(args.meta, "r", encoding="ascii") as f:
        meta = json.load(f)

    fps1, seg1_frames = read_video(args.segment1)
    fps2, seg2_frames = read_video(args.segment2)
    fps = fps1 if fps1 > 0 else fps2

    seg1_keep = meta["segment1_keep"]
    seg2_skip = meta["segment2_skip"]
    stitched = seg1_frames[:seg1_keep] + seg2_frames[seg2_skip:]
    write_video(args.output, fps, stitched)


def main():
    parser = argparse.ArgumentParser(description="Prepare and stitch segmented inversion-free runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input_video", required=True)
    prepare.add_argument("--edited_image", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--segment_frames", type=int, default=49)
    prepare.set_defaults(func=prepare_segments)

    stitch = subparsers.add_parser("stitch")
    stitch.add_argument("--segment1", required=True)
    stitch.add_argument("--segment2", required=True)
    stitch.add_argument("--meta", required=True)
    stitch.add_argument("--output", required=True)
    stitch.set_defaults(func=stitch_segments)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
