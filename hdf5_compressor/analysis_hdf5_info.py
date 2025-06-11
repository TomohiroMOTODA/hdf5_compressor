import h5py
import argparse
import os
import glob

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze HDF5 file for compression and chunking information.")
    parser.add_argument("--data_dir", type=str, help="Path to the HDF5 file to analyze.")
    args = parser.parse_args()

    is_shown = False
    data = glob.glob(os.path.join(args.data_dir, "*.hdf5"))    
    
    all_action_steps = 0

    for file_path in data:
        print(f"Analyzing file: {file_path}")
        with h5py.File(file_path, "r") as f:
            for name, dataset in f.items():
                if is_shown:
                    print(f"Dataset: {name}")
                    print(f"  Shape: {dataset.shape}")
                    print(f"  Dtype: {dataset.dtype}")
                    print(f"  Compression: {dataset.compression}")
                    print(f"  Compression options: {dataset.compression_opts}")
                    print(f"  Chunk shape: {dataset.chunks}")
                    print(f"  Size in bytes: {dataset.id.get_storage_size()}")

            action_shape = f['action'].shape
            all_action_steps += action_shape[0]
                
    print ("Total number of datasets:", len(data))
    print ("Total action steps across all datasets:", all_action_steps)
    print (f" total time: {all_action_steps / 50.} seconds")
    print (f" total action steps per dataset: {all_action_steps / 50. / 3600.0} hours")
    print (" action steps per dataset on average", all_action_steps / len(data))
    print("Analysis complete.")