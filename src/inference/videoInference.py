from src.utils.palette import inferenceColorPalette
from src.unet.architecture import Unet

import os
import cv2


import torch
from torchvision import transforms



def main(video_path, model_path):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (1280, 720))
        cv2.imshow('Video', frame_resized)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    video_path = os.path.join(base_dir, "data/tennisMatch/clips/clip1.mp4")
    model_path = os.path.join(base_dir, "models/unet_modelV2.pth")
    main(video_path=video_path, model_path=model_path)