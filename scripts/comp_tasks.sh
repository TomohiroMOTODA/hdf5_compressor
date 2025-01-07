#!/bin/bash

TARGET_FOLDER="/mnt/share/aist-aloha/train/004_8videos_grid/"

TASK_LIST=(
    "bluebox_delivery"
    "gear_task/01_washer_len700"
    "gear_task/02_gear_len500"
    "insert_usb_cable"
    "insert_usb_cable_fix_parts"
    "gear_task_003_01"
    "gear_task_003_02"
    "cable_routing_8pin_cross"
    "cable_routing_cross_narrow"
    "color_tape_grab_and_insert"
    "goods_grab_and_insert"
    "silver_pouch_001"
    "silver_pouch_002"
    "silver_pouch_004"
    "clear_zipper_bag_001"
    "clear_zipper_bag_reel_001"
    "clear_zipper_bag_reel_002"
    "receive_goods_and_add_to_basket"
    "tape_placed_in_random_places_long"
)

for t in "${TASK_LIST[@]}"; do
    dir="$TARGET_FOLDER$t"
    dir_name=$(basename "$dir")
    echo "Processing directory: $dir_name"

    python -m hdf5_compressor.compress_hdf5 --compress --dataset_dir ./data/sample --output_dir ./data --nproc 5 --quality 50 
    
    if [ $? -eq 0 ]; then
        echo "Data saved for $dir_name in $output_dir"
    else
        echo "Error processing $dir_name"
    fi
done

echo "All directories processed."