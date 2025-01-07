# Easy to Compress HDF5 data

## Usage

### Compress ALOHA trajectories
```console
python -m hdf5_compressor.compress_hdf5 --compress --dataset_dir <path to dataset> --output_dir <path to save compressed data> --nproc 5 --quality 50
```

### Compress ALOHA trajectries attached with Realsense Cameras
```console
python -m hdf5_compressor.compress_hdf5_rs --compress --dataset_dir <path to dataset> --output_dir <path to save compressed data> --nproc 5 --quality 50
```

### Visualize .hdf5
```console
python -m hdf5_compressor.visualize_hdf5 --compressed --dataset_dir <> --episode_idx 0
```