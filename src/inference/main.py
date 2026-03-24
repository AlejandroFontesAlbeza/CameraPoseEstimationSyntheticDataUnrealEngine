import cv2
import numpy as np


from inference.display_visualization import draw_lines_and_intersections, VideoInference
from inference.utils import get_device, get_model, get_tensor_transform, save_frame_info
from utils.dict_utils import inference_color_palette, intersections_lines, real_world_points
import config_inference
from inference.camera_pose import homography, camera_pose_estimation


def main():

    opencv_calibration = False
    show_mask = False
    resize = (1280,480)
    resize_no_mask = (640,480)

    device = get_device()

    model = get_model(config_inference.INFERENCE_MODEL_PATH, device)
    model.eval()

    transform = get_tensor_transform()

    video_inference = VideoInference(config_inference.INFERENCE_VIDEO_PATH, model,
                                    transform, device, inference_palette=inference_color_palette)
    width, height = video_inference.get_size()

    ### Calibration for camera intrinsic parameters, K

    if opencv_calibration:
        print("me falta implementar script")
        return
    else:
        "Manual calibration using FOV and image size"
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

    frame_number = 0
    for frame, predicted_mask_resized, predicted_mask_color in video_inference:
        frame_with_lines, img_intersections = draw_lines_and_intersections(
            frame.copy(), predicted_mask_resized, inference_color_palette, intersections_lines
        )

        H = homography(img_intersections, real_world_points)
        print("Homography matrix:\n", H)

        if H is not None:
            cam_position, cam_rotation = camera_pose_estimation(H, K)
            print(f"Camera Pos metres: {cam_position}")
            print(f"Camera Rot degrees: {cam_rotation}")
        else:
            print("Skipping Pose estimation due to insufficient intersections.")
            cam_position, cam_rotation = [0,0,0], [0,0,0]

        frame_with_mask = np.hstack((frame_with_lines, predicted_mask_color))
        save_frame_info(config_inference.OUTPUT_DIR, frame_number, frame_with_mask, cam_position, cam_rotation, FOV_x_deg)

        if show_mask:
            display = frame_with_mask
            display_resized = cv2.resize(display, resize)
        else:
            display = frame_with_lines
            display_resized = cv2.resize(display, resize_no_mask)

        cv2.imshow('Inference', display_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    video_inference.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
