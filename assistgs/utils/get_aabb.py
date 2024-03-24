'''
Author: error: git config user.name & please set dead value or install git && bigcileng@outlook.com
Date: 2024-02-29 15:39:20
LastEditors: jieyi-one && bigcileng@outlook.com
LastEditTime: 2024-02-29 22:20:16
FilePath: /assist-gs/scripts/get_aabb.py
Description: 

Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''
import numpy as np
import open3d as o3d
from pathlib import Path
from assistgs.utils.pcd2nerfstudio import create_ply_from_colmap
from tqdm import tqdm

def get_aabb(file_dir: Path = Path("/data/luoly/dataset/assist/530_scannet_table0_copy"), is_colmap_pcd: bool = False):
    bboxs = np.array([])
    pcd_files_dir = file_dir / "models" / "point_cloud"
    output_dir = file_dir / "bboxs_aabb.npy"
    pcd_files = sorted(pcd_files_dir.glob("*.ply"))
    for i, file in tqdm(enumerate(pcd_files)):
        instance_id = i+1
        pcd = o3d.io.read_point_cloud(str(file))
        if is_colmap_pcd:
            pcd = create_ply_from_colmap(pcd)
        aabb = pcd.get_minimal_oriented_bounding_box()
        xyz = np.asarray(aabb.center, dtype=np.float32)
        hwl = np.asarray(aabb.extent, dtype=np.float32)
        rotation = np.asarray(aabb.R, dtype=np.float32).flatten()
        bbox = np.concatenate([xyz, hwl, rotation, [instance_id]])
        bboxs = np.concatenate([bboxs, bbox])
    bboxs = np.reshape(bboxs, [-1, 16])
    assert bboxs.shape[0] == instance_id
    np.save(output_dir, bboxs)
    print(f"find {instance_id} objects, bboxs are saved in {output_dir}")

if __name__ == "__main__":
    get_aabb()