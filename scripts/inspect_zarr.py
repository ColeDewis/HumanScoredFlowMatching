import os
import sys

import zarr


def inspect_zarr(path):
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    try:
        # Open in read-only mode
        data = zarr.open(path, mode='r')
    except Exception as e:
        print(f"Could not open Zarr file: {e}")
        return

    print(f"\n{'='*20} ZARR INSPECTION {'='*20}")
    print(f"Path: {path}")
    
    # 1. Visual Tree Structure
    print("\n--- Hierarchy Tree ---")
    try:
        # This shows a nice ASCII tree of the entire file
        print(data.tree())
    except AttributeError:
        print("Root is a single Array (no tree structure).")

    # 2. Detailed Iteration
    print("\n--- Detailed Content ---")
    print(f"{'Path':<40} | {'Type':<10} | {'Shape':<20} | {'Dtype'}")
    print("-" * 85)

    def visitor(name, obj):
        # Determine if it is a Group or an Array
        is_group = isinstance(obj, zarr.hierarchy.Group)
        obj_type = "Group" if is_group else "Array"
        
        # Arrays have shapes and dtypes; Groups do not
        shape = str(obj.shape) if not is_group else "-"
        dtype = str(obj.dtype) if not is_group else "-"
        
        print(f"{name:<40} | {obj_type:<10} | {shape:<20} | {dtype}")

    if hasattr(data, 'visititems'):
        data.visititems(visitor)
    else:
        # Handle case where the root itself is just an array
        print(f"{'root':<40} | Array      | {data.shape:<20} | {data.dtype}")

    # 3. Attributes (Metadata)
    if len(data.attrs) > 0:
        print("\n--- Root Attributes (Metadata) ---")
        for key, val in data.attrs.items():
            print(f"{key}: {val}")

def print_episode_ends(zarr_path):
    try:
        z = zarr.open(zarr_path, mode='r')
        if 'meta/episode_ends' in z:
            print(z['meta/episode_ends'][:])
        else:
            print(f"Key 'meta/episode_ends' not found in {zarr_path}")
    except Exception as e:
        print(f"Error reading Zarr: {e}")

if __name__ == "__main__":
    # Get path from command line argument or default
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else "data.zarr"
    inspect_zarr(zarr_path)
    print_episode_ends(zarr_path)