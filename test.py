import cv2

SOURCE_VIDEO_PATH = "videos/match3.mp4"
TARGET_VIDEO_PATH = "videos/match3_result.mp4"

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    print("❌ OpenCV failed to open video")
else:
    print("✅ OpenCV successfully opened the video")
