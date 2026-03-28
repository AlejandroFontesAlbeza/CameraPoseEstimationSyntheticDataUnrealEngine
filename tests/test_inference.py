from inference.utils import get_model, get_device, get_tensor_transform
from inference.utils import VideoInference
import config_inference
from utils.dict_utils import inference_color_palette
from inference.camera_pose import homography, camera_pose_estimation

print("Running Inference Tests...")
def test_get_model():
    device = get_device()
    model = get_model(config_inference.INFERENCE_MODEL_PATH, device=device)
    assert model is not None

def test_frame_inference():
    device = get_device()
    model = get_model(config_inference.INFERENCE_MODEL_PATH, device=device)
    model.eval()
    transform = get_tensor_transform()
    video_inference = VideoInference(config_inference.INFERENCE_VIDEO_PATH, model,
                                    transform, device, inference_palette=inference_color_palette)
    frame, predicted_mask_resized, predicted_mask_color, inference_time, fps = next(video_inference)
    assert frame is not None
    assert predicted_mask_resized is not None
    assert predicted_mask_color is not None
    assert inference_time > 0
    assert fps > 0

def test_homography_and_pose_estimation():
    ## Simulate Data
    img_intersections = {
        1: [100, 200],
        2: [200, 200],
        3: [200, 300],
        4: [100, 300]
    }

    real_world_points = {
        1: [0, 0],
        2: [1, 0],
        3: [1, 1],
        4: [0, 1]
    }

    H = homography(img_intersections, real_world_points)
    assert H is not None
    assert H.shape == (3, 3)
    cx, cy = 320, 240
    cam_position, cam_rotation, f, FOV_x_deg = camera_pose_estimation(H, cx, cy)
    assert cam_position is not None
    assert cam_rotation is not None
    assert f > 0
    assert FOV_x_deg > 0

def test_inference_pipeline():
    device = get_device()
    model = get_model(config_inference.INFERENCE_MODEL_PATH, device=device)
    model.eval()
    transform = get_tensor_transform()
    video_inference = VideoInference(config_inference.INFERENCE_VIDEO_PATH, model,
                                    transform, device, inference_palette=inference_color_palette)
    frame, _, _, _, _ = next(video_inference)
    img_intersections = {
        1: [100, 200],
        2: [200, 200],
        3: [200, 300],
        4: [100, 300]
    }
    real_world_points = {
        1: [0, 0],
        2: [1, 0],
        3: [1, 1],
        4: [0, 1]
    }
    H = homography(img_intersections, real_world_points)
    assert H is not None
    cx, cy = frame.shape[1] / 2, frame.shape[0] / 2
    cam_position, cam_rotation, _, _ = camera_pose_estimation(H, cx, cy)
    assert cam_position is not None
    assert cam_rotation is not None


