import cv2
import numpy as np

import torch
from torchvision import transforms

from unet.unet import Unet
from config import INFERENCE_VIDEO_PATH, INFERENCE_MODEL_PATH
from utils.dict_utils import inference_color_palette
from inference.utils import VideoInference, draw_lines_and_intersections
from inference.camera_pose import homography, camera_pose_estimation
from utils.dict_utils import intersections_lines, real_world_points

def main():

    np.set_printoptions(precision=2, suppress=True) ##change numpy precision on prints

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet(in_channels=3, num_classes=10).to(device)
    model.load_state_dict(torch.load(INFERENCE_MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


    resize = (1280,480)
    resize_no_mask = (640,480)
    show_mask = True

    video_inference = VideoInference(INFERENCE_VIDEO_PATH, model,
                                    transform, device, inference_palette=inference_color_palette)
    width, height = video_inference.get_size()

    ### Calibration for camera intrinsic parameters, K

    FOV_x_deg = 51.282  # FOV horizontal
    FOV_x = np.deg2rad(FOV_x_deg)
    fx = (width / 2) / np.tan(FOV_x / 2)

    FOV_y = 2 * np.arctan((height / width) * np.tan(FOV_x / 2))
    fy = (height / 2) / np.tan(FOV_y / 2)

    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]], dtype=np.float32)

    for frame, predicted_mask_resized, predicted_mask_color in video_inference:
        frame_with_lines, img_intersections = draw_lines_and_intersections(
            frame.copy(), predicted_mask_resized, inference_color_palette, intersections_lines
        )

        H = homography(img_intersections, real_world_points)
        print("Homography matrix:\n", H)

        if H is not None:
            cam_position, cam_rotation = camera_pose_estimation(H, K)
            print(f"Camera Position (World Coordinates): {cam_position}")
            print(f"Camera Rotation (Euler angles in degrees): {cam_rotation}")
        else:
            print("Skipping Pose estimation due to insufficient intersections.")


        if show_mask:
            display = np.hstack((frame_with_lines, predicted_mask_color))
            display_resized = cv2.resize(display, resize)
        else:
            display = frame_with_lines
            display_resized = cv2.resize(display, resize_no_mask)

        cv2.imshow('Inference', display_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_inference.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
