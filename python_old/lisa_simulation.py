#from atmos_models import LISA
import open3d as o3d
# import struct
import numpy as np
import os

from tqdm import tqdm
from atmos_models import LISA

bin_path ='/home-local2/vikon4.extra.nobkp/mmdetection3d/data/kitti/training/velodyne'
dir_list = os.listdir(bin_path)
out_bin_path = '/home-local2/vikon4.extra.nobkp/mmdetection3d/data/kitti/training/velodyne_snow_75mm'


if not os.path.exists(out_bin_path):
    os.makedirs(out_bin_path)

out_dir_list = os.listdir(out_bin_path)

def write_pcd_to_bin(pcd, out_path: str):
    np.asarray(pcd).astype('float32').tofile(out_path)

def augment_point_cloud(dir_list):
    #Initialize LISA model
    lisa = LISA(atm_model='snow')
    
    for idx, file in enumerate(tqdm(dir_list)):
        if file in out_dir_list:
            continue
        else:
            file_path = os.path.join(bin_path, file)
            raw_lidar = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
            #this_pcd = bin_to_pcd(file_path)
            data_sim  = lisa.augment(raw_lidar,75) #snow_rate_75mm/hr
            out_path = os.path.join(out_bin_path, file)
            write_pcd_to_bin(data_sim, out_path)


dir_list = os.listdir(bin_path)
augment_point_cloud(dir_list)