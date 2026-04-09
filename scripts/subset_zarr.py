import argparse
import os

import numpy as np
import zarr


def subset_zarr(input_path: str, output_path: str, num_episodes: int):
    print(f"Opening source Zarr: {input_path}")
    src = zarr.open(input_path, mode='r')
    
    if 'meta/episode_ends' not in src:
        raise ValueError("Could not find 'meta/episode_ends'. Ensure this is a standard flattened Zarr format.")
    
    episode_ends = src['meta/episode_ends'][:]
    total_episodes = len(episode_ends)
    
    print(f"Dataset has {total_episodes} total episodes.")
    
    if num_episodes >= total_episodes:
        print(f"Warning: Requested {num_episodes} episodes, but only {total_episodes} exist. Copying the whole dataset.")
        num_episodes = total_episodes
        
    # Find the exact step index where our target episode ends
    step_limit = episode_ends[num_episodes - 1]
    print(f"Subsetting to the first {num_episodes} episodes (Total steps: {step_limit})")

    # Create the new destination Zarr
    dst = zarr.open(output_path, mode='w')

    def copy_recursively(src_group, dst_group):
        for key, item in src_group.items():
            if isinstance(item, zarr.hierarchy.Group):
                # If it's a folder (like 'data' or 'meta' or 'obs'), recreate it and step inside
                new_dst_group = dst_group.create_group(key)
                copy_recursively(item, new_dst_group)
            elif isinstance(item, zarr.core.Array):
                # If it's the episode_ends array, slice by episode count
                if key == 'episode_ends' and dst_group.name == '/meta':
                    data_slice = item[:num_episodes]
                # For all other data arrays (actions, images, states), slice by total step count
                else:
                    data_slice = item[:step_limit]

                # Recreate the array in the new Zarr with the same compression and chunks
                dst_group.array(
                    name=key,
                    data=data_slice,
                    chunks=item.chunks,
                    dtype=item.dtype,
                    compressor=item.compressor
                )
                print(f" -> Copied {dst_group.name}/{key}: shape {data_slice.shape}")

    # Kick off the recursive copy starting from the root
    copy_recursively(src, dst)
    print(f"\nSuccessfully created subset at: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # python subset_zarr.py --input data.zarr --output subset_data.zarr --n 50
    parser = argparse.ArgumentParser(description="Create a smaller subset of a trajectory Zarr.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original Zarr file")
    parser.add_argument("--output", type=str, required=True, help="Path for the new subset Zarr file")
    parser.add_argument("--n", type=int, required=True, help="Number of episodes to keep")
    
    args = parser.parse_args()
    
    if os.path.exists(args.output):
        print(f"Warning: Output path {args.output} already exists. It will be overwritten.")
        
    subset_zarr(args.input, args.output, args.n)