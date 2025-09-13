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
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.triangle_model import TriangleModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos

class Scene:

    triangles : TriangleModel

    def __init__(self, args : ModelParams, triangles : TriangleModel, init_opacity, init_size, nb_points, set_sigma, no_dome=False, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.triangles = triangles

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "vccsim_training_config.json")) and os.path.exists(os.path.join(args.source_path, "camera_info.json")):
            print("Found VCCSim configuration files, using VCCSim data set!")
            scene_info = sceneLoadTypeCallbacks["VCCSim"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.triangles.load(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)
                                                )
                                    )
        else:
            # Check configuration for mesh triangle initialization
            config_file = os.path.join(args.source_path, 'vccsim_training_config.json')
            use_mesh_triangles = False
            mesh_triangles_file = None
            mesh_opacity = init_opacity  # Default to the provided init_opacity
            
            if os.path.exists(config_file):
                try:
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        mesh_config = config.get('mesh', {})
                        use_mesh_triangles = mesh_config.get('use_mesh_triangles', False)
                        mesh_triangles_file = mesh_config.get('mesh_triangles_file', None)
                        mesh_opacity = mesh_config.get('mesh_opacity', init_opacity)
                        print(f"[DEBUG] Configuration loaded: use_mesh_triangles={use_mesh_triangles}")
                        print(f"[DEBUG] Mesh opacity override: {mesh_opacity}")
                        if mesh_triangles_file:
                            print(f"[DEBUG] Mesh triangles file: {mesh_triangles_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to load VCCSim configuration: {e}")
            
            if use_mesh_triangles and mesh_triangles_file and os.path.exists(mesh_triangles_file):
                print(f"Using direct mesh triangle initialization with opacity={mesh_opacity}")
                # Use preloaded mesh triangle data with high confidence
                self.triangles.create_from_mesh_triangles(
                    scene_info.point_cloud, self.cameras_extent, mesh_opacity, set_sigma, 
                    is_mesh_data=True)
            else:
                if use_mesh_triangles:
                    if not mesh_triangles_file:
                        print("[WARNING] Mesh triangle mode enabled but no mesh triangles file specified")
                    elif not os.path.exists(mesh_triangles_file):
                        print(f"[WARNING] Mesh triangles file not found: {mesh_triangles_file}")
                    
                    # Still use mesh triangle initialization method but with lower confidence
                    print(f"Using mesh triangle method with point cloud data (lower confidence) with opacity={mesh_opacity}")
                    self.triangles.create_from_mesh_triangles(
                        scene_info.point_cloud, self.cameras_extent, mesh_opacity, set_sigma,
                        is_mesh_data=False)
                else:
                    print("Using traditional point cloud initialization") 
                    self.triangles.create_from_pcd(scene_info.point_cloud, self.cameras_extent, init_opacity, init_size, nb_points, set_sigma, no_dome)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.triangles.save(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]