import os
import cv2
import argparse
from config import IMG_SIZE, VIDEO_FOLDER

def extract_frames_from_video(video_path, output_folder, num_frames=10, sample_every_k=5):
    """Extract frames and save as JPEG images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0
    
    # Create output folder for this video
    os.makedirs(output_folder, exist_ok=True)
    
    saved = 0
    if num_frames and num_frames > 0:
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                out_path = os.path.join(output_folder, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
    else:
        # Sequential sampling every k frames
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_every_k == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                out_path = os.path.join(output_folder, f"frame_{count:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
            count += 1
    
    cap.release()
    return saved

def extract_all_videos(input_root, output_root, num_frames=10, sample_every_k=5):
    """
    Recursively scan input_root for video files, extract frames into output_root
    preserving subfolder structure.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    for root, dirs, files in os.walk(input_root):
        for fname in files:
            if fname.lower().endswith(video_extensions):
                video_path = os.path.join(root, fname)
                # Build relative path from input_root
                rel_path = os.path.relpath(root, input_root)
                # Create corresponding output folder: output_root/rel_path/video_name_without_ext/
                video_name = os.path.splitext(fname)[0]
                output_subdir = os.path.join(output_root, rel_path, video_name)
                print(f"Extracting frames from: {video_path} -> {output_subdir}")
                saved = extract_frames_from_video(video_path, output_subdir, num_frames, sample_every_k)
                print(f"  Saved {saved} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=VIDEO_FOLDER, help="Root folder containing videos (with subfolders like Celeb-synthesis, etc.)")
    parser.add_argument("--output", type=str, default="extracted_frames", help="Root folder to save extracted frames")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames per video (uniform sampling)")
    parser.add_argument("--step", type=int, default=5, help="If num_frames is None, sample every K frames")
    args = parser.parse_args()
    
    import numpy as np  # needed for linspace
    extract_all_videos(args.input, args.output, args.num_frames, args.step)