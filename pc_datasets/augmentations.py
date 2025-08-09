import numpy as np
import open3d as o3d

import utils.pc_utils as pc_utils


def random_flip_yz(point_cloud, bboxes, p=0.5):
    # Flipping along the YZ plane
    if np.random.uniform() < p:
        point_cloud[:, 0] = -1 * point_cloud[:, 0]
        bboxes[:, 0] = -1 * bboxes[:, 0]
        bboxes[:, 6] = - bboxes[:, 6]
    return point_cloud, bboxes


def random_flip_xz(point_cloud, bboxes, p=0.5):
    # Flipping along the XZ plane
    if np.random.uniform() < p:
        point_cloud[:, 1] = -1 * point_cloud[:, 1]
        bboxes[:, 1] = -1 * bboxes[:, 1]
        bboxes[:, 6] = - bboxes[:, 6]
    return point_cloud, bboxes


def random_rotate(point_cloud, bboxes, angle_range=np.pi / 3):
    # Rotation along up-axis/Z-axis
    rot_angle = np.random.random() * angle_range - angle_range / 2
    rot_mat = pc_utils.rot_z(rot_angle)
    point_cloud[:, 0: 3] = np.dot(point_cloud[:, 0: 3], np.transpose(rot_mat))
    bboxes[:, 0: 3] = np.dot(bboxes[:, 0: 3], np.transpose(rot_mat))
    bboxes[:, 6] += rot_angle
    return point_cloud, bboxes


def random_scale(point_cloud, bboxes, use_height=False):
    # Augment point cloud scale: 0.85x-1.15x
    scale_ratio = np.random.random() * 0.3 + 0.85
    scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
    point_cloud[:, 0: 3] *= scale_ratio
    bboxes[:, 0: 3] *= scale_ratio
    bboxes[:, 3: 6] *= scale_ratio
    if use_height:
        point_cloud[:, -1] *= scale_ratio[0, 0]
    return point_cloud, bboxes


def color_augmentation(point_cloud, mean_color_rgb):
    rgb_color = point_cloud[:, 3:6] + mean_color_rgb
    rgb_color *= (1 + 0.4 * np.random.random(3) - 0.2)  # brightness change for each channel
    rgb_color += (0.1 * np.random.random(3) - 0.05)  # color shift for each channel
    rgb_color += np.expand_dims((0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1)  # jitter on each pixel
    rgb_color = np.clip(rgb_color, 0, 1)
    # randomly drop out 30% of the points' colors
    rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0]) > 0.3, -1)
    point_cloud[:, 3:6] = rgb_color - mean_color_rgb
    return point_cloud


def random_pick_point(min_xyz, max_xyz):
    # Randomly pick a point on the surface of a box
    surface_normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    surface_idx = np.random.randint(0, 6)
    normal = surface_normals[surface_idx]
    random_point_on_plane = np.random.random(3) * (max_xyz - min_xyz) + min_xyz
    random_point = min_xyz * (normal < 0) + max_xyz * (normal > 0) + random_point_on_plane * (normal == 0)
    return random_point


def density_aug(point_clouds, sampled_points):
    batch_size, num_points, _ = point_clouds.shape
    point_clouds_np = point_clouds.detach().cpu().numpy()
    if num_points >= sampled_points:
        choices = []
        for b, points in enumerate(point_clouds_np):
            x_min, y_min, z_min = np.min(points, axis=0)[: 3]
            x_max, y_max, z_max = np.max(points, axis=0)[: 3]
            basic_point = random_pick_point(np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max]))
            dist = np.sqrt(np.sum((points[:, :3] - basic_point) ** 2, axis=1))
            weights = (np.max(dist) / (dist + 1e-6)) ** 2
            weights /= np.sum(weights)
            choice = np.random.choice(num_points, sampled_points, replace=False, p=weights)
            choices.append(choice)
        return choices
    return None


# ==== VIRTUAL SCAN SIMULATION ====

def get_camera_candidate_locations(xyz):
    xyz_min, xyz_max = np.min(xyz, axis=0), np.max(xyz, axis=0)
    l, w = xyz_max[0] - xyz_min[0], xyz_max[1] - xyz_min[1]
    grid_size = min(l, w) / 50
    xy_map = np.zeros((int(l / grid_size) + 1, int(w / grid_size) + 1), dtype=np.uint8)
    norm_xy = np.floor_divide(xyz[:, :2] - xyz_min[:2], grid_size).astype(np.int64)
    xy_map[norm_xy[:, 0], norm_xy[:, 1]] = 1
    empty_locs_norm = np.where(xy_map == 0)
    empty_locs_norm = np.concatenate((empty_locs_norm[0].reshape(-1, 1), empty_locs_norm[1].reshape(-1, 1)), 1)
    camera_locs = empty_locs_norm * grid_size + xyz_min[:2]
    height = xyz_max[2] - xyz_min[2]
    camera_height = np.random.rand() * height / 2.0 + height / 2.0
    camera_locs = np.concatenate((camera_locs, np.full((camera_locs.shape[0], 1), camera_height)), 1)
    return camera_locs


def get_view_range_mask_fixed(_xyz_f, _camera_f):
    if _camera_f[2] > 0:
        visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] < _camera_f[2])
    else:
        visible_mask = (_xyz_f[..., 0] * _camera_f[0] + _xyz_f[..., 1] * _camera_f[1] <= (
            _camera_f[0] ** 2 + _camera_f[1] ** 2)) & (_xyz_f[..., 2] > _camera_f[2])
    return visible_mask


def get_wall_idx(xyz, eps=5e-2):
    xyz_min, xyz_max = np.min(xyz, axis=0), np.max(xyz, axis=0)
    x_wall_idx = (np.abs(xyz[..., 0] - xyz_min[0]) < eps) | (np.abs(xyz[..., 0] - xyz_max[0]) < eps)
    y_wall_idx = (np.abs(xyz[..., 1] - xyz_min[1]) < eps) | (np.abs(xyz[..., 1] - xyz_max[1]) < eps)
    wall_idx = x_wall_idx | y_wall_idx
    wall_xyz = xyz[wall_idx]
    return wall_xyz


def occlusion_simulation(xyz, label_mask, views=4, radius=1000):
    selected_mask = np.zeros(xyz.shape[0], dtype=bool)
    # ====  get candidate camera positions
    _xyz = xyz[label_mask != -1]
    camera_locs = get_camera_candidate_locations(_xyz)
    if camera_locs.shape[0] == 0:
        return np.ones_like(selected_mask, dtype=bool)
    selected_camera = 0
    _xyz_wall = get_wall_idx(xyz)
    try_times = 0
    while selected_camera < views:
        # ====  random select camera (view-point)
        idx = np.random.randint(camera_locs.shape[0])
        camera = camera_locs[idx]
        # ==== random select the interest point
        if _xyz_wall.shape[0] > 0:
            interest_point = _xyz_wall[np.random.choice(_xyz_wall.shape[0])]
        else:
            interest_point = np.array([0, 0, 0])
        _camera_f = camera - interest_point
        _xyz_f = xyz - interest_point
        # ==== determine the view range
        view_range_mask = get_view_range_mask_fixed(_xyz_f, _camera_f)
        to_select_idx = np.arange(xyz.shape[0])
        view_range_idx = to_select_idx[view_range_mask]
        if np.sum(view_range_mask) < 10:
            try_times += 1
            if try_times > np.maximum(5, views):
                return np.ones_like(selected_mask, dtype=bool)
            continue
        # ==== determine the visible points
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(_xyz_f[view_range_mask])
            _, pt_map = pcd.hidden_point_removal(_camera_f, radius)
        except Exception as e:
            print(e)
            return np.ones_like(selected_mask, dtype=bool)
        pt_map = np.array(pt_map)
        visible_idx = view_range_idx[pt_map]
        selected_mask[visible_idx] = True
        selected_camera += 1
    return selected_mask


def noise_simulation(xyz, jitter_value=2e-6):
    random_noise = (np.random.rand(xyz.shape[0], xyz.shape[1]) - 0.5) * jitter_value
    xyz += random_noise
    return xyz


def virtual_scan_simulation(point_cloud, label_mask, noise_jitter=0.01, view=4, radius=1000):
    # ===== occlusion simulation ======
    xyz = point_cloud[:, :3]
    selected_idx = occlusion_simulation(xyz, label_mask, views=view, radius=radius)
    # ===== noise simulation ======
    xyz = noise_simulation(xyz, jitter_value=noise_jitter)
    point_cloud[:, :3] = xyz
    point_cloud = point_cloud[selected_idx]
    return point_cloud
