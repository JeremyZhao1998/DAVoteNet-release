import os.path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def draw_point_cloud(input_points,
                     boxes3d=None,
                     point_marker='.',
                     point_size=1.2,
                     azim=-60,
                     save_name=None,
                     dpi=300):
    if not isinstance(input_points, np.ndarray):
        input_points = input_points.detach().cpu().numpy()
    x_min, y_min, z_min = np.min(input_points, axis=0)[: 3]
    x_max, y_max, z_max = np.max(input_points, axis=0)[: 3]
    x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min
    all_len = max(x_len, y_len, z_len)
    x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
    x_min, x_max = x_mid - all_len / 2, x_mid + all_len / 2
    y_min, y_max = y_mid - all_len / 2, y_mid + all_len / 2
    z_min, z_max = z_mid - all_len / 2, z_mid + all_len / 2
    # Plot axis
    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Turn off axis lines completely
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_color((0, 0, 0, 0))  # transparent
    # Turn off axis panes (background planes)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_zaxis.line.set_lw(0.)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=azim)
    ax.set_facecolor('dimgray')
    # Plot point clouds
    colors = input_points[:, 3:] if input_points.shape[1] >= 6 else 'b'
    ax.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], c=colors,
               marker=point_marker, s=point_size)
    # Plot boxes
    if boxes3d is not None:
        if not isinstance(boxes3d, np.ndarray):
            boxes3d = boxes3d.detach().cpu().numpy()
        for box3d in boxes3d:
            idx_list = [0, 1, 2, 3, 0]
            ax.plot(box3d[idx_list, 0], box3d[idx_list, 1], box3d[idx_list, 2], label='label', color='red', linewidth=2)
            idx_list = [4, 5, 6, 7, 4]
            ax.plot(box3d[idx_list, 0], box3d[idx_list, 1], box3d[idx_list, 2], label='label', color='red', linewidth=2)
            for i in range(4):
                idx_list = [i, i + 4]
                ax.plot(box3d[idx_list, 0], box3d[idx_list, 1], box3d[idx_list, 2], label='label', color='red', linewidth=2)
    if save_name is not None:
        plt.savefig(save_name, dpi=dpi)
        plt.close()
    else:
        plt.show()


def draw_gif(pc_np, save_path, save_name, boxes3d=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    images = []
    for i in tqdm(range(72), desc=f'generating gif to {save_path}'):
        degree = i * 5
        name = os.path.join(save_path, f'{save_name}_{degree}.png')
        draw_point_cloud(pc_np, boxes3d, azim=degree, save_name=name, dpi=100)
        images.append(Image.open(name))
    images[0].save(os.path.join(save_path, f'{save_name}.gif'),
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
    os.system(f"rm {save_path}/*.png")
    pass


def export_ply(pc_np, boxes3d=None, save_name=None):
    """
    Export point cloud to PLY with color.
    pc_np shape: (N, 6), columns = x, y, z, r, g, b
    """
    N = pc_np.shape[0]
    header = f'''ply
    format ascii 1.0
    element vertex {N}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open('test.ply' if save_name is None else save_name, 'w') as f:
        f.write(header)
        for p in pc_np:
            # 颜色需要保证是整数
            r = int(p[3] * 255)
            g = int(p[4] * 255)
            b = int(p[5] * 255)
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")
    pass
