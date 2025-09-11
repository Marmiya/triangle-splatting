#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.triangle_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, use_parallel=None, num_workers=4):
    """
    Load PLY file with automatic optimization based on file size
    Args:
        path: PLY file path
        use_parallel: Force parallel loading (None=auto, True=parallel, False=single-thread)
        num_workers: Number of parallel workers for processing
    """
    print(f"Loading PLY file: {path}")
    import time
    start_time = time.time()
    
    # Check file size to decide loading strategy
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"PLY file size: {file_size_mb:.1f} MB")
    
    # Auto-decide parallel strategy for large files
    if use_parallel is None:
        use_parallel = file_size_mb > 50.0  # Use parallel for files > 50MB
    
    if use_parallel:
        print(f"Using parallel loading with {num_workers} workers for large file")
        return _fetchPlyParallel(path, num_workers)
    else:
        print("Using single-threaded loading")
        return _fetchPlySingleThread(path)


def _fetchPlySingleThread(path):
    """Optimized single-threaded PLY loading"""
    import time
    start_time = time.time()
    
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    num_vertices = len(vertices)
    
    print(f"PLY file loaded in {time.time() - start_time:.2f}s, processing {num_vertices} vertices...")
    process_start = time.time()
    
    # Pre-allocate arrays for better performance
    positions = np.empty((num_vertices, 3), dtype=np.float32)
    colors = np.empty((num_vertices, 3), dtype=np.float32)
    normals = np.empty((num_vertices, 3), dtype=np.float32)
    
    # Direct array assignment (faster than vstack)
    positions[:, 0] = vertices['x']
    positions[:, 1] = vertices['y']
    positions[:, 2] = vertices['z']
    
    colors[:, 0] = vertices['red'] / 255.0
    colors[:, 1] = vertices['green'] / 255.0
    colors[:, 2] = vertices['blue'] / 255.0
    
    normals[:, 0] = vertices['nx']
    normals[:, 1] = vertices['ny']
    normals[:, 2] = vertices['nz']
    
    print(f"Data processing completed in {time.time() - process_start:.2f}s")
    print(f"Total PLY loading time: {time.time() - start_time:.2f}s")
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def _fetchPlyParallel(path, num_workers=4):
    """Parallel PLY loading for large files"""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    start_time = time.time()
    
    # Load PLY data (still single-threaded due to plyfile limitation)
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    num_vertices = len(vertices)
    
    load_time = time.time() - start_time
    print(f"PLY file loaded in {load_time:.2f}s, processing {num_vertices} vertices with {num_workers} workers...")
    
    # Process data in parallel chunks
    process_start = time.time()
    chunk_size = max(1000, num_vertices // num_workers)  # Minimum chunk size of 1000
    
    def process_chunk(start_idx, end_idx):
        """Process a chunk of vertices"""
        chunk_vertices = vertices[start_idx:end_idx]
        chunk_size = end_idx - start_idx
        
        positions = np.empty((chunk_size, 3), dtype=np.float32)
        colors = np.empty((chunk_size, 3), dtype=np.float32)
        normals = np.empty((chunk_size, 3), dtype=np.float32)
        
        positions[:, 0] = chunk_vertices['x']
        positions[:, 1] = chunk_vertices['y']
        positions[:, 2] = chunk_vertices['z']
        
        colors[:, 0] = chunk_vertices['red'] / 255.0
        colors[:, 1] = chunk_vertices['green'] / 255.0
        colors[:, 2] = chunk_vertices['blue'] / 255.0
        
        normals[:, 0] = chunk_vertices['nx']
        normals[:, 1] = chunk_vertices['ny']
        normals[:, 2] = chunk_vertices['nz']
        
        return start_idx, positions, colors, normals
    
    # Create processing tasks
    tasks = []
    for i in range(0, num_vertices, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, num_vertices)
        tasks.append((start_idx, end_idx))
    
    print(f"Processing {len(tasks)} chunks in parallel...")
    
    # Execute tasks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(process_chunk, start, end): (start, end)
            for start, end in tasks
        }
        
        for future in as_completed(future_to_task):
            start, end = future_to_task[future]
            try:
                start_idx, positions, colors, normals = future.result()
                results.append((start_idx, positions, colors, normals))
                print(f"Processed chunk [{start}:{end}] ({end-start} vertices)")
            except Exception as e:
                print(f"Error processing chunk [{start}:{end}]: {e}")
                raise
    
    # Sort results by start index and concatenate
    results.sort(key=lambda x: x[0])
    
    print("Concatenating parallel results...")
    all_positions = np.concatenate([r[1] for r in results], axis=0)
    all_colors = np.concatenate([r[2] for r in results], axis=0)
    all_normals = np.concatenate([r[3] for r in results], axis=0)
    
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    
    print(f"Parallel processing completed in {process_time:.2f}s")
    print(f"Total PLY loading time: {total_time:.2f}s")
    print(f"Performance: {num_vertices / total_time:.0f} vertices/second")
    
    return BasicPointCloud(points=all_positions, colors=all_colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readVCCSimCameras(camera_infos, images_folder):
    """
    Read VCCSim camera data directly from camera_info.json - no pose file needed
    The C++ VCCSimDataConverter has already processed poses into camera_info.json
    """
    cam_infos = []
    
    for idx, camera_info in enumerate(camera_infos):
        
        # Get pose data directly from camera_info (already processed by VCCSimDataConverter)
        if 'rotation' not in camera_info:
            raise ValueError(f"Camera {idx}: Missing required 'rotation' parameter")
        if 'translation' not in camera_info:
            raise ValueError(f"Camera {idx}: Missing required 'translation' parameter")
        
        rotation = camera_info['rotation']  # [qx, qy, qz, qw]
        translation = camera_info['translation']  # [x, y, z]
        
        # VCCSim stores Camera to World (C2W) format: camera position and rotation in world coordinates
        # But Triangle Splatting expects World to Camera (W2C) format (same as COLMAP)
        # Need to convert C2W to W2C by matrix inversion
        
        # Convert quaternion to rotation matrix (C2W rotation)
        # VCCSim stores as [qx, qy, qz, qw], qvec2rotmat expects [qw, qx, qy, qz] (COLMAP format)
        qvec_vccsim = np.array(rotation)  # [qx, qy, qz, qw]
        qvec_colmap = np.array([qvec_vccsim[3], qvec_vccsim[0], qvec_vccsim[1], qvec_vccsim[2]])  # [qw, qx, qy, qz]
        R_c2w = qvec2rotmat(qvec_colmap)  # Camera to World rotation matrix
        t_c2w = np.array(translation)     # Camera to World translation (camera position in world)
        
        # Convert C2W to W2C by matrix inversion
        # For a transformation matrix [R t; 0 1], the inverse is [R^T -R^T*t; 0 1]
        R_w2c = R_c2w.T                   # W2C rotation = transpose of C2W rotation
        t_w2c = -R_w2c @ t_c2w            # W2C translation = -R^T * camera_position
        
        # Apply the same transpose as COLMAP processing for consistency
        R = np.transpose(R_w2c)
        T = t_w2c
        
        # Get camera parameters (already processed by VCCSimDataConverter)
        if 'width' not in camera_info:
            raise ValueError(f"Camera {idx}: Missing required 'width' parameter")
        if 'height' not in camera_info:
            raise ValueError(f"Camera {idx}: Missing required 'height' parameter")
        if 'focal_x' not in camera_info:
            raise ValueError(f"Camera {idx}: Missing required 'focal_x' parameter")
        
        width = camera_info['width']
        height = camera_info['height']
        focal_x = camera_info['focal_x']
        focal_y = camera_info.get('focal_y', focal_x)  # focal_y can fallback to focal_x for square pixels
        
        # Calculate FOV
        FovX = focal2fov(focal_x, width)
        FovY = focal2fov(focal_y, height)
        
        # Get image info - support both jpg and png
        image_name = camera_info.get('image_name', f'image_{idx:06d}.jpg')  # Default to jpg
        image_path = camera_info.get('image_path', os.path.join(images_folder, image_name))
        
        # If image_path from camera_info doesn't exist, try in images_folder
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, image_name)
        
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                # Convert to RGB if needed (handles various formats)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                print(f"Warning: Error loading image {image_path}: {e}, creating dummy image")
                image = Image.new('RGB', (width, height), color=(128, 128, 128))
        else:
            # Create dummy image if not found
            print(f"Warning: Image not found: {image_path}, creating dummy image")
            image = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        uid = camera_info.get('uid', idx)
        
        cam_infos.append(CameraInfo(
            uid=uid, 
            R=R, 
            T=T, 
            FovY=FovY, 
            FovX=FovX, 
            image=image,
            image_path=image_path, 
            image_name=image_name, 
            width=width, 
            height=height
        ))
    
    return cam_infos

def readVCCSimSceneInfo(config_path, images=None, eval=False):
    """
    Read VCCSim scene data directly from VCCSim configuration files
    Args:
        config_path: Path to VCCSim config directory containing camera_info.json and PLY file
        images: Images directory path (can be None, will use image_path from camera_info)
        eval: Whether to split train/test cameras
    """
    print("Reading VCCSim Scene Info")
    
    # Load VCCSim configuration (optional - mainly for image_directory fallback)
    config_file = os.path.join(config_path, 'vccsim_training_config.json')
    vccsim_config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            vccsim_config = json.load(f)
    
    # Load camera info (contains all pose and camera data)
    camera_info_file = os.path.join(config_path, 'camera_info.json')
    if not os.path.exists(camera_info_file):
        raise FileNotFoundError(f"Camera info file not found: {camera_info_file}")
        
    with open(camera_info_file, 'r') as f:
        camera_infos = json.load(f)
    
    print(f"Loaded {len(camera_infos)} cameras from camera_info.json")
    
    # Determine images folder (fallback only if image_path in camera_info is not absolute)
    if images is None:
        images_folder = vccsim_config.get('image_directory', '')
    else:
        images_folder = images
    
    # Read cameras - pose data already processed by VCCSimDataConverter
    print("Reading VCCSim Cameras")
    cam_infos_unsorted = readVCCSimCameras(camera_infos, images_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    # Split train/test if eval
    if eval:
        # Use every 8th image for testing (similar to COLMAP default)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    print(f"Split cameras: {len(train_cam_infos)} training, {len(test_cam_infos)} test")
    
    # Calculate normalization
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Check for mesh triangles first, then fallback to point cloud
    mesh_triangles_file = os.path.join(config_path, 'mesh_triangles.ply')
    init_points_file = os.path.join(config_path, 'init_points.ply')
    
    if os.path.exists(mesh_triangles_file):
        # Use mesh triangles for initialization
        ply_file = mesh_triangles_file
        try:
            pcd = fetchPly(ply_file)
            print(f"Loaded mesh triangles with {len(pcd.points)} triangle vertices from {ply_file}")
        except Exception as e:
            print(f"Error loading mesh triangles PLY file: {e}")
            raise
    elif os.path.exists(init_points_file):
        # Fallback to traditional point cloud
        ply_file = init_points_file
        try:
            pcd = fetchPly(ply_file)
            print(f"Loaded point cloud with {len(pcd.points)} points from {ply_file}")
        except Exception as e:
            print(f"Error loading point cloud PLY file: {e}")
            raise
    else:
        raise FileNotFoundError(f"No initialization data found: neither {mesh_triangles_file} nor {init_points_file} exists")
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_file
    )
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "VCCSim": readVCCSimSceneInfo
}