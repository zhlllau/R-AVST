import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

sys.path.append("/hdd0/zl/VideoChat-R1-AVQA-Audio/src/open_r1")

from my_qwen_utils_fps025 import fetch_video

csv_path = "/hdd0/zl/Grounded-SAM-2/1.CaptionAnalysis/train_final_subject_caption_analysis.csv"
video_dir = "/hdd0/zl/Grounded-SAM-2/2.Videos"
output_dir = "/hdd0/zl/VideoChat-R1-AVQA-Audio/VideoNPY_1_FPS/train"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
video_names = df["video_name"].unique()

for video_name in tqdm(video_names, desc="Generating .npy files from CSV"):
    video_path = os.path.join(video_dir, video_name) + ".mp4"
    video_id = os.path.splitext(video_name)[0]
    npy_path = os.path.join(output_dir, f"{video_id}.npy")

    if not os.path.exists(video_path):
        print(f"[Warning] Video not found: {video_path}")
        continue

    if os.path.exists(npy_path):
        continue 

    try:
        video_tensor = fetch_video({"video": video_path})
        np.save(npy_path, video_tensor.numpy())
    except Exception as e:
        print(f"[Error] Failed to process {video_name}: {e}")
