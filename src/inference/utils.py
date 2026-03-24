import os
import cv2
import json


import torch
from unet.unet import Unet
from torchvision import transforms


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def get_model(model_path, device):
    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
    return model

def get_tensor_transform():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform

def save_frame_info(output_dir, frame_number, frame_with_mask, cam_position, cam_rotation, FOV=90):
    frames_dir = os.path.join(output_dir, "frames")
    jsons_dir = os.path.join(output_dir, "jsons")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    frame_path = os.path.join(frames_dir, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_path, frame_with_mask)

    pose_data = {
        "frame_number": frame_number,
        "camera_position": cam_position.tolist(),
        "camera_rotation": cam_rotation.tolist(),
        "FOV": FOV
    }
    pose_path = os.path.join(jsons_dir, f"pose_{frame_number:04d}.json")
    with open(pose_path, 'w') as f:
        json.dump(pose_data, f, indent=4)

    print(f"Saved frame to {frame_path} and pose info to {pose_path}")
