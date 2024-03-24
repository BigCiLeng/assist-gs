import torch
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
def create_ply_from_colmap(pcd, applied_transform=None):
    """Writes a ply file from colmap.

    Args:
        filename: file name for .ply
        recon_dir: Directory to grab colmap points
        output_dir: Directory to output .ply
    """
    applied_transform = torch.tensor([[0.0,1.0,0.0,0.0],
                                      [1.0,0.0,0.0,0.0],
                                      [-0.0,-0.0,-1.0,-0.0]
                                      ])
    
    # Load point Positions
    points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
    if applied_transform is not None:
        assert applied_transform.shape == (3, 4)
        points3D = torch.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]

    # Load point colours
    points3D = o3d.utility.Vector3dVector(points3D.cpu().numpy())
    pcd.points = points3D
    return pcd

def colmap_test():
    # root_dir = Path("/data/luoly/dataset/assist/530_scannet_table0_copy/models/point_cloud")
    # colmap_pcd_dir = root_dir / "colmap"
    # files = colmap_pcd_dir.glob("*.ply")
    # for file in tqdm(files):
    #     pcd = o3d.io.read_point_cloud(str(file))
    #     pcd = colmapPcd2nerfstudio(pcd)
    #     o3d.io.write_point_cloud(str(root_dir / file.name), pcd)
    file = "/data/luoly/dataset/assist/530_scannet_table0_copy/colmap_dense_pc.ply"
    pcd = o3d.io.read_point_cloud(str(file))
    pcd = create_ply_from_colmap(pcd)
    o3d.io.write_point_cloud("/data/luoly/dataset/assist/530_scannet_table0_copy/dense_pc.ply", pcd)

if __name__ == "__main__":
    colmap_test()