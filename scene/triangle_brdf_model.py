"""
Triangle BRDF Model (Standalone Implementation)

BRDF-based Triangle Splatting for physically-based rendering
Completely independent implementation without TriangleModel inheritance

Key Features:
- BRDF material parameters only (no Spherical Harmonics)
- Minimal memory footprint
- Full training and densification support
- UE5-compatible materials
- Direct initialization from MVS mesh triangles
"""

import torch
import numpy as np
from torch import nn
import os
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.sh_utils import SH2RGB


class TriangleBRDFModel(nn.Module):
    """
    Standalone BRDF Triangle Model

    Parameters per triangle:
    - Geometry: triangles_points [N, 3, 3], opacity [N, 1], sigma [N, 1]
    - Materials: base_color [N, 3], roughness [N, 1], metallic [N, 1]

    No SH parameters, minimal memory footprint.
    """

    def __init__(self):
        """Initialize BRDF model (sh_degree parameter removed)"""
        super().__init__()

        # Geometry parameters
        self._triangles_points = torch.empty(0)
        self._opacity = torch.empty(0)
        self._sigma = torch.empty(0)
        self._mask = torch.empty(0)

        # BRDF material parameters
        self._base_color = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)

        # Training state
        self.max_radii2D = torch.empty(0)
        self.max_density_factor = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        # Densification tracking
        self.max_scaling = torch.empty(0)
        self._num_points_per_triangle = torch.empty(0)
        self._cumsum_of_points_per_triangle = torch.empty(0)
        self._number_of_points = 0

        # Training parameters
        self.split_size = 0
        self.triangle_area = 0
        self.image_size = 0
        self.importance_score = 0
        self.nb_points = 0
        self.large = False

        # Activation functions
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.exponential_activation = lambda x: torch.exp(x)
        self.inverse_exponential_activation = lambda x: torch.log(x)

    # ========================================================================
    # Property Accessors
    # ========================================================================

    @property
    def get_triangles_points(self):
        return self._triangles_points

    @property
    def get_triangles_points_flatten(self):
        return self._triangles_points.flatten(0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_base_color(self):
        return torch.sigmoid(self._base_color)

    @property
    def get_roughness(self):
        return torch.sigmoid(self._roughness)

    @property
    def get_metallic(self):
        return torch.sigmoid(self._metallic)

    @property
    def get_num_points_per_triangle(self):
        return self._num_points_per_triangle

    @property
    def get_cumsum_of_points_per_triangle(self):
        return self._cumsum_of_points_per_triangle

    @property
    def get_number_of_points(self):
        return self._number_of_points

    @property
    def get_max_scaling(self):
        return self.max_scaling

    # ========================================================================
    # Initialization from MVS Mesh Triangles
    # ========================================================================

    def create_from_mesh_triangles(self, preloaded_pcd: BasicPointCloud, spatial_lr_scale: float,
                                   init_opacity: float, set_sigma: float, is_mesh_data: bool = True):
        """
        Initialize from pre-loaded MVS mesh triangles

        Args:
            preloaded_pcd: BasicPointCloud containing mesh triangle vertices
                          Expected format: points [N*3, 3] reshaped to [N, 3, 3]
                          colors [N, 3] for per-triangle colors
            spatial_lr_scale: Spatial learning rate scale
            init_opacity: Initial opacity value (typically 0.9-1.0 for mesh)
            set_sigma: Initial sigma value for window function
            is_mesh_data: True if data from mesh (higher confidence)
        """
        self.spatial_lr_scale = spatial_lr_scale
        self.nb_points = 3

        triangles_np = np.asarray(preloaded_pcd.points).reshape(-1, 3, 3)
        colors_np_raw = np.asarray(preloaded_pcd.colors)

        # Handle both per-triangle and per-vertex colors
        if colors_np_raw.shape[0] == triangles_np.shape[0]:
            # Already per-triangle colors
            colors_np = colors_np_raw
        else:
            # Per-vertex colors - average to get per-triangle colors
            colors_np = colors_np_raw.reshape(-1, 3, 3).mean(axis=1)

        num_triangles = triangles_np.shape[0]

        self.large = num_triangles > 100000

        # Initialize geometry
        triangles_torch = torch.from_numpy(triangles_np).float().cuda()
        self._triangles_points = nn.Parameter(triangles_torch.requires_grad_(True))

        opacities = inverse_sigmoid(
            init_opacity * torch.ones((num_triangles, 1), dtype=torch.float32, device="cuda")
        )
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        sigmas = self.inverse_exponential_activation(
            set_sigma * torch.ones((num_triangles, 1), dtype=torch.float32, device="cuda")
        )
        self._sigma = nn.Parameter(sigmas.requires_grad_(True))

        self._mask = nn.Parameter(
            torch.ones((num_triangles, 1), dtype=torch.float32, device="cuda").requires_grad_(True)
        )

        # Initialize BRDF from colors
        rgb_colors = torch.from_numpy(colors_np).float().cuda()
        rgb_colors = torch.clamp(rgb_colors, 0.01, 0.99)

        self._base_color = nn.Parameter(inverse_sigmoid(rgb_colors).requires_grad_(True))
        self._roughness = nn.Parameter(
            inverse_sigmoid(0.5 * torch.ones((num_triangles, 1), device="cuda")).requires_grad_(True)
        )
        self._metallic = nn.Parameter(
            inverse_sigmoid(0.05 * torch.ones((num_triangles, 1), device="cuda")).requires_grad_(True)
        )

        # Initialize tracking
        self.max_radii2D = torch.zeros((num_triangles), device="cuda")
        self.max_density_factor = torch.zeros((num_triangles), device="cuda")
        self.max_scaling = torch.zeros(num_triangles, device="cuda")

        num_points_list = [3] * num_triangles
        self._num_points_per_triangle = torch.tensor(num_points_list, dtype=torch.int, device='cuda')
        self._cumsum_of_points_per_triangle = torch.cumsum(
            torch.nn.functional.pad(self._num_points_per_triangle, (1, 0), value=0), 0, dtype=torch.int
        )[:-1]
        self._number_of_points = num_triangles

    # ========================================================================
    # Training Setup
    # ========================================================================

    def training_setup(self, training_args, lr_features, lr_opacity,
                      lr_sigma, lr_triangles_points_init):
        """Setup optimizer for BRDF training"""
        self.denom = torch.zeros((self.get_triangles_points.shape[0], 1), device="cuda")

        self.split_size = training_args.split_size
        self.lr_sigma = lr_sigma
        self.start_lr_sigma = training_args.start_lr_sigma
        self.max_noise_factor = training_args.max_noise_factor
        self.add_shape = training_args.add_shape

        param_groups = [
            {'params': [self._base_color], 'lr': lr_features, "name": "base_color"},
            {'params': [self._roughness], 'lr': lr_features * 0.5, "name": "roughness"},
            {'params': [self._metallic], 'lr': lr_features * 0.5, "name": "metallic"},
            {'params': [self._opacity], 'lr': lr_opacity, "name": "opacity"},
            {'params': [self._triangles_points], 'lr': lr_triangles_points_init, "name": "triangles_points"},
            {'params': [self._sigma], 'lr': lr_sigma, "name": "sigma"},
            {'params': [self._mask], 'lr': 0.00001, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(
            lr_init=lr_triangles_points_init,
            lr_final=lr_triangles_points_init / 100,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        """Update learning rate for triangle points"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "triangles_points":
                lr = self.triangle_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def oneupSHdegree(self):
        """Compatibility stub for training loop (no SH in BRDF model)"""
        pass

    # ========================================================================
    # Densification Support
    # ========================================================================

    def densification_postfix(self, new_triangles_points, new_base_color, new_roughness,
                             new_metallic, new_opacities, new_sigma, new_mask):
        """Add new triangles to model"""
        d = {
            "triangles_points": new_triangles_points,
            "base_color": new_base_color,
            "roughness": new_roughness,
            "metallic": new_metallic,
            "opacity": new_opacities,
            "sigma": new_sigma,
            "mask": new_mask
        }

        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        self._triangles_points = optimizable_tensors["triangles_points"]
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        # Reset tracking
        current_count = self.get_triangles_points.shape[0]
        self.denom = torch.zeros((current_count, 1), device="cuda")
        self.max_radii2D = torch.zeros((current_count), device="cuda")
        self.max_density_factor = torch.zeros((current_count), device="cuda")

        # Extend statistics
        old_count = len(self.triangle_area) if hasattr(self, 'triangle_area') and self.triangle_area is not None else 0
        if old_count > 0 and old_count < current_count:
            new_triangle_area = torch.zeros(current_count, device="cuda")
            new_triangle_area[:old_count] = self.triangle_area[:old_count]
            self.triangle_area = new_triangle_area
        elif old_count == 0:
            self.triangle_area = torch.zeros(current_count, device="cuda")

        self.max_scaling = torch.cat((self.max_scaling, torch.zeros(new_opacities.shape[0], device="cuda")), dim=0)

        # Update point tracking
        num_points_list = [3] * current_count
        self._num_points_per_triangle = torch.tensor(num_points_list, dtype=torch.int, device='cuda')
        self._cumsum_of_points_per_triangle = torch.cumsum(
            torch.nn.functional.pad(self._num_points_per_triangle, (1, 0), value=0), 0, dtype=torch.int
        )[:-1]
        self._number_of_points = current_count

    def _cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate new tensors to optimizer state"""
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                    )

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """Remove triangles based on mask"""
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._triangles_points = optimizable_tensors["triangles_points"]
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_density_factor = self.max_density_factor[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_scaling = self.max_scaling[valid_points_mask]

        # Update tracking
        current_count = self.get_triangles_points.shape[0]
        num_points_list = [3] * current_count
        self._num_points_per_triangle = torch.tensor(num_points_list, dtype=torch.int, device='cuda')
        self._cumsum_of_points_per_triangle = torch.cumsum(
            torch.nn.functional.pad(self._num_points_per_triangle, (1, 0), value=0), 0, dtype=torch.int
        )[:-1]
        self._number_of_points = current_count

    def _prune_optimizer(self, mask):
        """Prune optimizer state"""
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ========================================================================
    # Save and Load
    # ========================================================================

    def save(self, path):
        """Save model"""
        mkdir_p(path)

        state_dict = {
            "triangles_points": self._triangles_points,
            "sigma": self._sigma,
            "opacity": self._opacity,
            "base_color": self._base_color,
            "roughness": self._roughness,
            "metallic": self._metallic,
            "mask": self._mask,
        }

        torch.save(state_dict, os.path.join(path, 'brdf_model.pt'))

        hyperparameters = {
            "max_radii2D": self.max_radii2D,
            "denom": self.denom,
            "spatial_lr_scale": self.spatial_lr_scale,
            "num_points_per_triangle": self._num_points_per_triangle,
            "cumsum_of_points_per_triangle": self._cumsum_of_points_per_triangle,
            "number_of_points": self._number_of_points,
            "max_scaling": self.max_scaling,
            "max_density_factor": self.max_density_factor,
        }

        torch.save(hyperparameters, os.path.join(path, 'hyperparameters.pt'))

    def load(self, path):
        """Load model"""
        state_dict = torch.load(os.path.join(path, 'brdf_model.pt'))

        num_triangles = state_dict["triangles_points"].shape[0]

        self._triangles_points = state_dict["triangles_points"].cuda().float().requires_grad_(True)
        self._sigma = state_dict["sigma"].cuda().float().requires_grad_(True)
        self._opacity = state_dict["opacity"].cuda().float().requires_grad_(True)
        self._base_color = state_dict["base_color"].cuda().float().requires_grad_(True)
        self._roughness = state_dict["roughness"].cuda().float().requires_grad_(True)
        self._metallic = state_dict["metallic"].cuda().float().requires_grad_(True)
        self._mask = state_dict["mask"].cuda().float().requires_grad_(True)

        # Load hyperparameters
        hyperparams = torch.load(os.path.join(path, 'hyperparameters.pt'))
        self.max_radii2D = hyperparams["max_radii2D"]
        self.denom = hyperparams["denom"]
        self.spatial_lr_scale = hyperparams["spatial_lr_scale"]
        self._num_points_per_triangle = hyperparams["num_points_per_triangle"]
        self._cumsum_of_points_per_triangle = hyperparams["cumsum_of_points_per_triangle"]
        self._number_of_points = hyperparams["number_of_points"]
        self.max_scaling = hyperparams["max_scaling"]
        self.max_density_factor = hyperparams["max_density_factor"]

    def capture(self):
        """Capture state for checkpointing"""
        return (
            self._base_color,
            self._roughness,
            self._metallic,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """Restore from checkpoint"""
        (self._base_color,
         self._roughness,
         self._metallic,
         self._opacity,
         self.max_radii2D,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args

        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_brdf_statistics(self):
        """Get BRDF parameter statistics"""
        stats = {
            "base_color": {
                "min": self.get_base_color.detach().min(dim=0).values.cpu().numpy(),
                "max": self.get_base_color.detach().max(dim=0).values.cpu().numpy(),
                "mean": self.get_base_color.detach().mean(dim=0).cpu().numpy(),
                "std": self.get_base_color.detach().std(dim=0).cpu().numpy(),
            },
            "roughness": {
                "min": self.get_roughness.detach().min().item(),
                "max": self.get_roughness.detach().max().item(),
                "mean": self.get_roughness.detach().mean().item(),
                "std": self.get_roughness.detach().std().item(),
            },
            "metallic": {
                "min": self.get_metallic.detach().min().item(),
                "max": self.get_metallic.detach().max().item(),
                "mean": self.get_metallic.detach().mean().item(),
                "std": self.get_metallic.detach().std().item(),
            },
        }
        return stats

    def print_brdf_statistics(self):
        """Print BRDF statistics"""
        stats = self.get_brdf_statistics()

        print("\n" + "="*80)
        print("BRDF Parameter Statistics")
        print("="*80)

        print("\nBase Color (RGB):")
        print(f"  Min:  [{stats['base_color']['min'][0]:.3f}, {stats['base_color']['min'][1]:.3f}, {stats['base_color']['min'][2]:.3f}]")
        print(f"  Max:  [{stats['base_color']['max'][0]:.3f}, {stats['base_color']['max'][1]:.3f}, {stats['base_color']['max'][2]:.3f}]")
        print(f"  Mean: [{stats['base_color']['mean'][0]:.3f}, {stats['base_color']['mean'][1]:.3f}, {stats['base_color']['mean'][2]:.3f}]")

        print("\nRoughness:")
        print(f"  Range: [{stats['roughness']['min']:.3f}, {stats['roughness']['max']:.3f}]")
        print(f"  Mean:  {stats['roughness']['mean']:.3f} ± {stats['roughness']['std']:.3f}")

        print("\nMetallic:")
        print(f"  Range: [{stats['metallic']['min']:.3f}, {stats['metallic']['max']:.3f}]")
        print(f"  Mean:  {stats['metallic']['mean']:.3f} ± {stats['metallic']['std']:.3f}")

        print("="*80 + "\n")

    # ========================================================================
    # Densification Helper Methods (adapted from TriangleModel)
    # ========================================================================

    def _update_params(self, selected_indices):
        """Split selected triangles into 4 sub-triangles (BRDF version)"""
        selected_triangles_points = self._triangles_points[selected_indices]

        A = selected_triangles_points[:, 0, :]
        B = selected_triangles_points[:, 1, :]
        C = selected_triangles_points[:, 2, :]

        M_AB = (A + B) / 2
        M_AC = (A + C) / 2
        M_BC = (B + C) / 2

        sub1 = torch.stack([A, M_AB, M_AC], dim=1)
        sub2 = torch.stack([B, M_AB, M_BC], dim=1)
        sub3 = torch.stack([C, M_AC, M_BC], dim=1)
        sub4 = torch.stack([M_AB, M_AC, M_BC], dim=1)

        new_triangles_points = torch.cat([sub1, sub2, sub3, sub4], dim=0)

        new_base_color = self._base_color[selected_indices].repeat(4, 1)
        new_roughness = self._roughness[selected_indices].repeat(4, 1)
        new_metallic = self._metallic[selected_indices].repeat(4, 1)
        new_opacities = self._opacity[selected_indices].repeat(4, 1)
        new_sigma = self._sigma[selected_indices].repeat(4, 1)
        new_mask = torch.ones_like(self._mask[selected_indices].repeat(4, 1))

        return new_triangles_points, new_base_color, new_roughness, new_metallic, new_opacities, new_sigma, new_mask

    def _update_params_small(self, idxs):
        """Clone and add noise to small triangles (BRDF version)"""
        new_triangles_points = self._triangles_points[idxs]
        n = new_triangles_points.shape[0]

        v1 = new_triangles_points[:, 1] - new_triangles_points[:, 0]
        v2 = new_triangles_points[:, 2] - new_triangles_points[:, 0]
        normals = torch.cross(v1, v2, dim=1)
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-9)

        min_coords = new_triangles_points.min(dim=1).values
        max_coords = new_triangles_points.max(dim=1).values
        shape_sizes = max_coords - min_coords

        max_noise_factor = self.max_noise_factor
        noise_scale = shape_sizes * max_noise_factor
        noise = (torch.rand(n, 1, 3, device=new_triangles_points.device) - 0.5) * noise_scale.unsqueeze(1)

        dot_products = (noise * normals.unsqueeze(1)).sum(dim=-1, keepdim=True)
        noise_in_plane = noise - dot_products * normals.unsqueeze(1)

        new_triangles_points_noisy = new_triangles_points + noise_in_plane

        opacity_old = self.get_opacity[idxs]
        opacity_new = inverse_sigmoid(1.0 - torch.pow(1.0 - opacity_old, 1.0 / 2))

        return (torch.cat([self._triangles_points[idxs], new_triangles_points_noisy], dim=0),
                torch.cat([self._base_color[idxs], self._base_color[idxs]], dim=0),
                torch.cat([self._roughness[idxs], self._roughness[idxs]], dim=0),
                torch.cat([self._metallic[idxs], self._metallic[idxs]], dim=0),
                torch.cat([opacity_new, opacity_new], dim=0),
                torch.cat([self._sigma[idxs], self._sigma[idxs]], dim=0),
                torch.cat([self._mask[idxs], torch.ones((n, 1), device=self._mask.device)], dim=0))

    def _sample_alives(self, probs, num, big_mask, alive_indices=None):
        """Sample triangles for densification"""
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)

        num_available = (probs > 0).sum().item()
        num_to_sample = min(num, num_available)

        if num_to_sample <= 0:
            return torch.empty(0, dtype=torch.long, device=probs.device)

        sampled_idxs = torch.multinomial(probs, num_to_sample, replacement=False)

        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]

        device = probs.device
        cost_true = torch.tensor(3, dtype=torch.int64, device=device)
        cost_false = torch.tensor(1, dtype=torch.int64, device=device)
        costs = torch.where(big_mask[sampled_idxs], cost_true, cost_false)

        cum_costs = torch.cumsum(costs, dim=0)
        valid_mask = cum_costs <= num
        sampled_idxs = sampled_idxs[valid_mask]

        return sampled_idxs

    def replace_tensors_to_optimizer(self, inds=None):
        """Replace optimizer tensors (BRDF version)"""
        tensors_dict = {
            "triangles_points": self._triangles_points,
            "base_color": self._base_color,
            "roughness": self._roughness,
            "metallic": self._metallic,
            "opacity": self._opacity,
            "sigma": self._sigma,
            "mask": self._mask
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._triangles_points = optimizable_tensors["triangles_points"]
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

    def add_new_gs(self, cap_max, oddGroup=True, dead_mask=None):
        """Add new triangles through split/clone (BRDF version)"""
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(self.add_shape * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        num_gs += dead_mask.sum() if dead_mask is not None else 0

        if num_gs <= 0:
            return 0

        if oddGroup:
            probs = self.get_opacity.squeeze(-1)
        else:
            eps = torch.finfo(torch.float32).eps
            probs = self.get_sigma.squeeze(-1)
            probs = 1 / (probs + eps)

        if dead_mask is not None:
            probs[dead_mask] = 0

        compar = self.image_size
        big_mask = compar > self.split_size

        add_idx = self._sample_alives(probs=probs, num=num_gs, big_mask=big_mask)

        # If no triangles can be sampled, skip densification
        if add_idx.numel() == 0:
            return 0

        big_mask = compar[add_idx] > self.split_size
        small_mask = ~big_mask
        big_indices = add_idx[big_mask]
        small_indices = add_idx[small_mask]

        num_big = big_indices.shape[0]
        if num_big > 0:
            (split_triangles_points,
             split_base_color,
             split_roughness,
             split_metallic,
             split_opacity,
             split_sigma,
             split_mask) = self._update_params(big_indices)
        else:
            split_triangles_points = torch.empty((0, 3, 3), device=self._triangles_points.device)
            split_base_color = torch.empty((0, 3), device=self._base_color.device)
            split_roughness = torch.empty((0, 1), device=self._roughness.device)
            split_metallic = torch.empty((0, 1), device=self._metallic.device)
            split_opacity = torch.empty((0, 1), device=self._opacity.device)
            split_sigma = torch.empty((0, 1), device=self._sigma.device)
            split_mask = torch.empty((0, 1), device=self._mask.device)

        num_small = small_indices.shape[0]
        if num_small > 0:
            (clone_triangles_points,
             clone_base_color,
             clone_roughness,
             clone_metallic,
             clone_opacity,
             clone_sigma,
             clone_mask) = self._update_params_small(small_indices)
        else:
            clone_triangles_points = torch.empty((0, 3, 3), device=self._triangles_points.device)
            clone_base_color = torch.empty((0, 3), device=self._base_color.device)
            clone_roughness = torch.empty((0, 1), device=self._roughness.device)
            clone_metallic = torch.empty((0, 1), device=self._metallic.device)
            clone_opacity = torch.empty((0, 1), device=self._opacity.device)
            clone_sigma = torch.empty((0, 1), device=self._sigma.device)
            clone_mask = torch.empty((0, 1), device=self._mask.device)

        new_triangles_points = torch.cat([split_triangles_points, clone_triangles_points], dim=0)
        new_base_color = torch.cat([split_base_color, clone_base_color], dim=0)
        new_roughness = torch.cat([split_roughness, clone_roughness], dim=0)
        new_metallic = torch.cat([split_metallic, clone_metallic], dim=0)
        new_opacity = torch.cat([split_opacity, clone_opacity], dim=0)
        new_sigma = torch.cat([split_sigma, clone_sigma], dim=0)
        new_mask = torch.cat([split_mask, clone_mask], dim=0)

        self.densification_postfix(new_triangles_points, new_base_color, new_roughness,
                                   new_metallic, new_opacity, new_sigma, new_mask)
        self.replace_tensors_to_optimizer(inds=add_idx)

        mask = torch.zeros(self._opacity.shape[0], dtype=torch.bool, device="cuda")
        mask[add_idx] = True
        if dead_mask is not None:
            mask[torch.nonzero(dead_mask, as_tuple=True)] = True
        self.prune_points(mask)

    def remove_final_points(self, mask):
        """Remove triangles based on mask (BRDF version)"""
        self.prune_points(mask)

        current_count = self.get_triangles_points.shape[0]
        old_count = len(self.triangle_area) if hasattr(self, 'triangle_area') and self.triangle_area is not None else 0

        if old_count > 0:
            new_triangle_area = torch.zeros(current_count, device="cuda")
            new_triangle_area[:min(old_count, current_count)] = self.triangle_area[:min(old_count, current_count)]
            self.triangle_area = new_triangle_area

            new_image_size = torch.zeros(current_count, device="cuda")
            new_image_size[:min(old_count, current_count)] = self.image_size[:min(old_count, current_count)]
            self.image_size = new_image_size

            new_importance_score = torch.zeros(current_count, device="cuda")
            new_importance_score[:min(old_count, current_count)] = self.importance_score[:min(old_count, current_count)]
            self.importance_score = new_importance_score
        else:
            self.triangle_area = torch.zeros(current_count, device="cuda")
            self.image_size = torch.zeros(current_count, device="cuda")
            self.importance_score = torch.zeros(current_count, device="cuda")
