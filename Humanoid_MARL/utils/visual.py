from typing import Union, List
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2
from datetime import datetime

from Humanoid_MARL import PACKAGE_ROOT, ROOT


def save_video(frames: List[np.array],
               path: str = "data/ppo",
               name: str = "ppo_train_video.mp4",
               fps: int = 30):
    """
    Save a sequence of frames as a video.

    Parameters:
    - frames: List of NumPy arrays representing frames. Each array should have shape (height, width, 3).
    - path: Path to the directory where the video will be saved.
    - name: Name of the video file.
    - fps: Frames per second for the video.
    """
    if not frames:
        raise ValueError("Input list should contain at least one frame.")

    height, width, _ = frames[0].shape
    size = (width, height)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"{timestamp}_{name}"

    if path:
        SAVE_PATH = os.path.join(ROOT, path, timestamped_name)
    else:
        SAVE_PATH = os.path.join("../", PACKAGE_ROOT, path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    video_writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, size)
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

# def save_video(rgb_array: List[np.array],
#                path: str = "data/ppo",
#                name: str = "ppo_train_video.mp4"):
#     """
#     Save a sequence of RGB frames as a video.

#     Parameters:
#     - rgb_array: List of NumPy arrays representing RGB frames. Each array should have shape (height, width, 3).
#     - path: Path to the directory where the video will be saved.
#     - name: Name of the video file.
#     """
#     if not rgb_array:
#         raise ValueError("Input list should contain at least one RGB frame.")

#     fig, ax = plt.subplots()
#     ax.axis('off')

#     def update(frame):
#         ax.imshow(frame)

#     ani = FuncAnimation(fig, update, frames=rgb_array, interval=100, repeat=False)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     timestamped_name = f"{timestamp}_{name}"

#     if path:
#         SAVE_PATH = os.path.join(ROOT, path, timestamped_name)
#     else:
#         SAVE_PATH = os.path.join("../", PACKAGE_ROOT, path)

#     # ani.save(SAVE_PATH, writer='ffmpeg', fps=30)
#     breakpoint()
#     ani.save(SAVE_PATH, writer='ffmpeg', fps=30, codec='libx264')
#     plt.close(fig)

def save_rgb_image(rgb_array : np.array,
                    path : str = "data/ppo",
                    name : str = "ppo_train_img.png") -> None:
    """
    Plot an RGB image represented as a NumPy array.

    Parameters:
    - rgb_array: NumPy array representing the RGB image. It should have shape (height, width, 3).
    """
    if rgb_array.shape[2] != 3:
        raise ValueError("Input array should have shape (height, width, 3) for RGB image.")
    plt.imshow(rgb_array)
    plt.axis('off')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"{timestamp}_{name}"
    if path:
        SAVE_PATH = os.path.join(ROOT, path, timestamped_name)
        plt.savefig(SAVE_PATH, bbox_inches='tight', pad_inches=0)
    else:
        SAVE_PATH = os.path.join("../", PACKAGE_ROOT, path)