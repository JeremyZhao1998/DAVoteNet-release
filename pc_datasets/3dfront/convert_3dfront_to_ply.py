import trimesh
import numpy as np
from tqdm import tqdm
import os
import open3d as o3d

save_dir = './'
dir_path = './'

obj_files = os.listdir(dir_path)
for obj_file in tqdm(obj_files):
    if not obj_file.endswith('.glb'):
        continue
    scene = trimesh.load_mesh(os.path.join(dir_path, obj_file))
    all_sampled_points = []
    all_sampled_colors = []

    # num_samples_per_geometry = 100000
    num_scene = 100000
    all_areas = []
    for geometry in scene.geometry.values():
        vertices = geometry.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)

        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        depth = max_coords[2] - min_coords[2]

        bbox_surface_area = 2 * (width * height + height * depth + depth * width)
        all_areas.append(bbox_surface_area)
    total_sum = sum(all_areas)
    ratios = [num / total_sum for num in all_areas]

    for idx, geometry in enumerate(scene.geometry.values()):
        vertex_colors = geometry.visual.vertex_colors
        face_colors = geometry.visual.face_colors

        sample_num = int(num_scene * ratios[idx])
        sampled_points, face_indices = geometry.sample(sample_num, return_index=True)

        sampled_colors = face_colors[face_indices, :3]

        all_sampled_points.append(sampled_points)
        all_sampled_colors.append(sampled_colors)

    all_sampled_points = np.vstack(all_sampled_points)
    all_sampled_colors = np.vstack(all_sampled_colors)

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(all_sampled_points)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(all_sampled_colors / 255.0)

    file_name = obj_file.replace('.glb', '.ply')
    o3d.io.write_point_cloud(os.path.join(save_dir, file_name), point_cloud_o3d)
