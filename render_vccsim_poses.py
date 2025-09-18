#
# VCCSim Pose-based Rendering Script
# Renders a sequence of images defined by a VCCSim poses.txt file
# Based on triangle-splatting render.py with VCCSim pose integration
#

import torch
from scene import Scene
from scene.triangle_model import TriangleModel
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from utils.general_utils import safe_state, PILtoTorch
from utils.graphics_utils import focal2fov, getWorld2View2
from utils.camera_utils import loadCam
from arguments import ModelParams, PipelineParams
from PIL import Image

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Path to the trained Triangle Splatting model
MODEL_PATH = r"C:\UEProjects\VCCSimDev\Saved\RatSplatting\TriangleSplatting\RatSplatting\session_20250912_035702_mesh"

# Path to VCCSim poses.txt file (format: timestamp x y z qx qy qz qw)
POSES_FILE = r"C:\UEProjects\VCCSimDev\Saved\pose_ue.txt"

# Output image dimensions (focal lengths will be computed from training config)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Output directory for rendered images
OUTPUT_DIR = os.path.join(MODEL_PATH, "rendered_sequence")

# Rendering parameters
ITERATION = -1  # Use -1 for latest iteration, or specify exact iteration number
WHITE_BACKGROUND = False  # Set to True for white background, False for black
RESOLUTION_SCALE = 1.0  # 1.0 = full resolution, 0.5 = half resolution, etc.

# =============================================================================
# VCCSim POSE PROCESSING
# =============================================================================

def parse_vccsim_pose_line(line):
    """Parse a single line from VCCSim poses.txt file"""
    values = line.strip().split()
    if len(values) != 8:
        raise ValueError(f"Invalid pose line format (expected 8 values, got {len(values)}): {line}")

    timestamp = float(values[0])
    location = np.array([float(values[1]), float(values[2]), float(values[3])])
    quaternion = np.array([float(values[4]), float(values[5]), float(values[6]), float(values[7])])  # [qx, qy, qz, qw]

    return timestamp, location, quaternion

def convert_vccsim_coordinates(location):
    """
    Convert VCCSim left-handed coordinates to Triangle Splatting right-handed coordinates
    Coordinate transformation: UE(X_forward, Y_right, Z_up) -> RightHanded(X_right, Y_forward, Z_up)
    """
    return np.array([
        0.01 * location[1],  # X_rh = Y_ue (right direction)
        0.01 * location[0],  # Y_rh = X_ue (forward direction)
        0.01 * location[2]   # Z_rh = Z_ue (up direction unchanged)
    ])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix"""
    qx, qy, qz, qw = q

    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])

    return R

def load_training_config(model_path):
    """Load camera parameters from training configuration"""
    config_file = os.path.join(model_path, "config", "vccsim_training_config.json")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Training config not found: {config_file}")

    import json
    with open(config_file, 'r') as f:
        config = json.load(f)

    camera_config = config.get('camera', {})

    return {
        'original_width': camera_config.get('width', 1920),
        'original_height': camera_config.get('height', 1080),
        'original_focal_x': camera_config.get('focal_length_x', 1000.0),
        'original_focal_y': camera_config.get('focal_length_y', 1000.0),
        'fov_degrees': camera_config.get('fov_degrees', 60.0)
    }

def compute_scaled_focal_lengths(original_config, target_width, target_height):
    """Compute focal lengths for target resolution based on training config"""
    scale_x = target_width / original_config['original_width']
    scale_y = target_height / original_config['original_height']

    focal_x = original_config['original_focal_x'] * scale_x
    focal_y = original_config['original_focal_y'] * scale_y

    return focal_x, focal_y

def convert_vccsim_pose_to_camera(timestamp, location, quaternion, width, height, focal_x, focal_y):
    """
    Convert VCCSim pose to Triangle Splatting camera format
    VCCSim poses represent Camera-to-World transformation
    Triangle Splatting expects World-to-Camera transformation (like COLMAP)
    """

    # Convert coordinates from left-handed to right-handed
    converted_location = convert_vccsim_coordinates(location)

    # Convert quaternion to rotation matrix (Camera-to-World)
    R_c2w = quaternion_to_rotation_matrix(quaternion)

    # Apply coordinate system transformation to rotation
    # Coordinate swap matrix: [0 1 0; 1 0 0; 0 0 1] (swap X and Y axes)
    coord_transform = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    R_c2w_converted = coord_transform @ R_c2w @ coord_transform.T

    # Convert C2W to W2C by matrix inversion
    # For transformation matrix [R t; 0 1], inverse is [R^T -R^T*t; 0 1]
    R_w2c = R_c2w_converted.T  # W2C rotation = transpose of C2W rotation
    t_w2c = -R_w2c @ converted_location  # W2C translation = -R^T * camera_position

    # Apply the same transpose as COLMAP processing for consistency with Triangle Splatting
    R = np.transpose(R_w2c)
    T = t_w2c

    # Calculate FOV from focal lengths
    FovX = focal2fov(focal_x, width)
    FovY = focal2fov(focal_y, height)

    return R, T, FovX, FovY

def load_vccsim_poses(poses_file):
    """Load and parse VCCSim poses.txt file"""
    poses = []

    print(f"Loading VCCSim poses from: {poses_file}")

    with open(poses_file, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                timestamp, location, quaternion = parse_vccsim_pose_line(line)
                poses.append((timestamp, location, quaternion))
            except ValueError as e:
                print(f"Warning: Skipping invalid pose at line {line_num + 1}: {e}")

    print(f"Loaded {len(poses)} poses")
    return poses

def create_camera_infos_from_poses(poses, width, height, focal_x, focal_y):
    """Create CameraInfo objects from VCCSim poses"""
    cam_infos = []

    for idx, (timestamp, location, quaternion) in enumerate(poses):
        # Convert pose to camera parameters
        R, T, FovX, FovY = convert_vccsim_pose_to_camera(
            timestamp, location, quaternion, width, height, focal_x, focal_y
        )

        # Create a minimal dummy image (we only need rendering output)
        dummy_image = Image.new('RGB', (width, height), color=(0, 0, 0))

        # Create camera info
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=dummy_image,
            image_path=f"render_{idx:06d}.png",
            image_name=f"render_{idx:06d}",
            width=width,
            height=height
        )

        cam_infos.append(cam_info)

    return cam_infos

# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================

def render_pose_sequence(model_path, poses_file, output_dir, width, height,
                        iteration=-1, white_background=False, resolution_scale=1.0):
    """
    Render a sequence of images from VCCSim poses
    """

    # Ensure output directory exists
    makedirs(output_dir, exist_ok=True)

    # Load training configuration to get camera parameters
    print("Loading training configuration...")
    training_config = load_training_config(model_path)

    # Compute scaled focal lengths based on target resolution
    focal_x, focal_y = compute_scaled_focal_lengths(training_config, width, height)

    print(f"Training resolution: {training_config['original_width']}x{training_config['original_height']}")
    print(f"Target resolution: {width}x{height}")
    print(f"Original focal lengths: fx={training_config['original_focal_x']:.2f}, fy={training_config['original_focal_y']:.2f}")
    print(f"Scaled focal lengths: fx={focal_x:.2f}, fy={focal_y:.2f}")

    # Load poses
    poses = load_vccsim_poses(poses_file)
    if not poses:
        raise ValueError("No valid poses found in poses file")

    # Create camera infos from poses
    cam_infos = create_camera_infos_from_poses(poses, width, height, focal_x, focal_y)

    print(f"Rendering {len(cam_infos)} views to {output_dir}")

    # Initialize Triangle Splatting model
    with torch.no_grad():
        # Create triangle model
        triangles = TriangleModel(sh_degree=3)  # Default SH degree

        # Load trained model
        if iteration == -1:
            from utils.system_utils import searchForMaxIteration
            iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

        if iteration is None:
            raise ValueError(f"No trained model found in {model_path}")

        print(f"Loading trained model at iteration {iteration}")
        triangles.load(os.path.join(model_path, "point_cloud", f"iteration_{iteration}"))

        # Set background color
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Create proper pipeline parameters with all required attributes
        pipeline = PipelineParams.__new__(PipelineParams)
        pipeline.convert_SHs_python = False
        pipeline.compute_cov3D_python = False
        pipeline.depth_ratio = 1.0
        pipeline.debug = False

        # Create argument-like object for camera loading
        class SimpleArgs:
            def __init__(self):
                self.data_device = "cuda"
                self.resolution = 1  # Use scale 1 to avoid auto-scaling warnings

        args = SimpleArgs()

        # Render each pose
        for idx, cam_info in enumerate(tqdm(cam_infos, desc="Rendering poses")):
            # Create camera object
            camera = loadCam(args, idx, cam_info, resolution_scale)

            # Render the view
            rendering = render(camera, triangles, pipeline, background)["render"]

            # Save rendered image
            output_path = os.path.join(output_dir, f"render_{idx:06d}.png")
            torchvision.utils.save_image(rendering, output_path)

        print(f"Rendering completed! {len(cam_infos)} images saved to {output_dir}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize system state (RNG)
    safe_state(False)

    print("VCCSim Pose-based Rendering Script")
    print("==================================")
    print(f"Model path: {MODEL_PATH}")
    print(f"Poses file: {POSES_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Iteration: {ITERATION}")
    print(f"White background: {WHITE_BACKGROUND}")
    print(f"Resolution scale: {RESOLUTION_SCALE}")
    print()

    # Validate inputs
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    if not os.path.exists(POSES_FILE):
        raise FileNotFoundError(f"Poses file not found: {POSES_FILE}")

    # Run rendering
    try:
        render_pose_sequence(
            model_path=MODEL_PATH,
            poses_file=POSES_FILE,
            output_dir=OUTPUT_DIR,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            iteration=ITERATION,
            white_background=WHITE_BACKGROUND,
            resolution_scale=RESOLUTION_SCALE
        )
    except Exception as e:
        print(f"Error during rendering: {e}")
        raise