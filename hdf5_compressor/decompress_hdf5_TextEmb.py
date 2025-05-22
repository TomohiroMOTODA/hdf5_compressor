#
import os
import time
import cv2
import h5py
import glob
import argparse
from multiprocessing import Pool

## CONSTANTS
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"]
DCAMERA_NAMES = ["dcam_high", "dcam_low"]
##

def load_compressed_hdf5(info):
    dataset_path, output_dir = info
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    dst_path = os.path.join(output_dir, os.path.basename(dataset_path))

    with h5py.File(dataset_path, 'r') as root,  h5py.File(dst_path, 'w') as dst:

        def recursive_copy(src_group, dst_group):
            for name, item in src_group.items():
                if isinstance(item, h5py.Group):
                    dst_subgroup = dst_group.create_group(name)
                    recursive_copy(item, dst_subgroup)
                elif isinstance(item, h5py.Dataset):
                    dst_group.create_dataset(name, data=item[()])
            for key, value in src_group.attrs.items():
                dst_group.attrs[key] = value

        recursive_copy(root, dst)

        action = root['/action'][()]
        max_timesteps = len(action)

        image_dict = dict()
        data_dict = dict()
        for cam_name in CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'] = []

        for cam_name in root[f'/observations/images/'].keys():
            emc_images = root[f'/observations/images/{cam_name}'][()]
            image_dict[cam_name] = list()
            for img in emc_images:
                decompressed_image = cv2.imdecode(img , 1)
                image_dict[cam_name].append(decompressed_image)

            del dst[f'/observations/images/{cam_name}']
            _ = dst.create_dataset(f'/observations/images/{cam_name}', (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]

        for name , array in data_dict.items():
            dst[name][...] = array

def save_hdf5(file_info):
    raise NotImplementedError("This function is not implemented in this script.")

def main(args):
    dataset_dir   = args["dataset_dir"]
    nproc         = args["nproc"]
    output_dir    = args["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    episodes = glob.glob(os.path.join(dataset_dir, '*.hdf5'), recursive=True)
    print (f'Load:{dataset_dir}')
    print (f'Size: {len(episodes)} episodes')

    pool = Pool(nproc)
    pool.map(load_compressed_hdf5, [(episode, output_dir) for episode in  episodes])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--output_dir', action='store', type=str, default="./data", help='Dataset dir.')
    parser.add_argument('--nproc', type=int, default=1, help='number of processes', required=False)  
    main (vars(parser.parse_args()))