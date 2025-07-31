from ultralytics import YOLO
from utils import read_video, save_video
from detection import start_detection
import cv2
import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

if __name__ == "__main__":
    PLAYER_DETECTION_MODEL = YOLO("models/best.pt")
    SOURCE_VIDEO_PATH = "videos/match3.mp4"
    TARGET_VIDEO_PATH = "videos/match3_result.mp4"

    video_frames = read_video(SOURCE_VIDEO_PATH)

    # Start the detection process
    print("Starting detection...")
    start_detection(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, PLAYER_DETECTION_MODEL)
    print("Detection completed. Results saved to:", TARGET_VIDEO_PATH)


