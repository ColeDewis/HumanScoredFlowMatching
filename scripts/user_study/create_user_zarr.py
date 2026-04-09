# Basically the same as the merge_sirius_zarrs file, just with a bit of extra syntax baked in.
import argparse

import numpy as np
import zarr
from termcolor import cprint

# !!!!! IMPORTANT !!!!!
# THIS IS CALCULATING WEIGHTS ASSUMING THAT WE DO NOT HAVE ENOUGH CORRECTION TRAJECTORIES
# TO BE ABLE TO DO THE WEIGHTING TO 50% FOR THEM
# THUS, WE INSTEAD GIVE THE MAX POSSIBLE WEIGHT TO CORRECTIONS, LEAVING DEMOS THE SAME

def combine_and_weight_sirius(input_paths, output_path, demo_paths):
    """
    input_paths: List of all zarr paths to combine.
    demo_paths: List of paths that are 'pure' demonstrations (N=60).
    """
    # 1. Scanning
    total_steps = 0
    for p in input_paths:
        total_steps += zarr.open(p, mode='r')['data/action'].shape[0]

    # 2. Create Output
    out_z = zarr.open_group(output_path, mode='w')
    out_data = out_z.create_group('data')
    out_meta = out_z.create_group('meta')
    
    # 3. Merge Logic with Class Tracking
    step_offset = 0
    ep_offset = 0
    
    # Masks to track the three Sirius classes across the whole dataset
    is_demo_file = np.zeros(total_steps, dtype=bool)
    is_takeover = np.zeros(total_steps, dtype=bool)

    for p in input_paths:
        z = zarr.open(p, mode='r')
        n = z['data/action'].shape[0]
        curr_slice = slice(step_offset, step_offset + n)
        
        # Track if this is a demo file or an intervention rollout file
        if p in demo_paths:
            is_demo_file[curr_slice] = True
        
        # Copy standard data (action, state, img, etc.)
        for key in z['data'].keys():
            if key == 'takeovers': 
                is_takeover[curr_slice] = z['data/takeovers'][:]
                continue
            
            # Create dataset on first pass
            if key not in out_data:
                shape = (total_steps,) + z[f'data/{key}'].shape[1:]
                out_data.create_dataset(key, shape=shape, dtype=z[f'data/{key}'].dtype, chunks=z[f'data/{key}'].chunks)
            
            out_data[key][curr_slice] = z[f'data/{key}'][:]
        
        # Meta: Shift episode ends
        if 'episode_ends' not in out_meta:
            total_eps = sum([zarr.open(f, 'r')['meta/episode_ends'].size for f in input_paths])
            out_meta.create_dataset('episode_ends', shape=(total_eps,), dtype='int64')
            
        n_ep = z['meta/episode_ends'].size
        out_meta['episode_ends'][ep_offset : ep_offset + n_ep] = z['meta/episode_ends'][:] + step_offset
        
        step_offset += n
        ep_offset += n_ep

    # 4. CALCULATE WEIGHTS (The Sirius Way)
    weights = np.zeros(total_steps, dtype=np.float32)

    # Class A: Pure Demos (Keep original importance)
    weights[is_demo_file] = 1.0

    # Class B: Human Corrections (Interventions)
    num_intv_steps = np.sum(is_takeover)
    if num_intv_steps > 0:
        rollout_file_mask = ~is_demo_file
        total_rollout_steps = np.sum(rollout_file_mask)
        weights[is_takeover] = total_rollout_steps / num_intv_steps

    # Class C: Autonomous Robot Steps are left at 0.0

    # --- SAVE TO ZARR ---
    # Save the weights
    out_data.create_dataset('traj_weight', data=weights, dtype='float32', chunks=(100,), overwrite=True)
    
    # Save the raw takeovers mask for future flexibility
    out_data.create_dataset('takeovers', data=is_takeover.astype(np.float32), dtype='float32', chunks=(100,), overwrite=True)
    
    cprint(f"Final Weights: Demos=1.0, Interventions={weights.max():.2f}, Robot=0.0", "cyan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and weight Zarr files for Sirius.")
    parser.add_argument("--user", type=str, required=True, help="Identifier for the user who collected the data (e.g., coled_01)")
    parser.add_argument("--round", type=int, required=True, help="Round 1 or 2")
    parser.add_argument("--multiple", action='store_true', help="Set for multiple intervention protocol")
    
    args = parser.parse_args()
    user_id = args.user
    round = args.round
    multiple = args.multiple

    protocol_folder = "multiple" if multiple else "single"

    if round == 1:
        all_files = ["/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/base.zarr", 
                    f"/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/{user_id}/{protocol_folder}/round1.zarr"]
        demo_files = ["/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/base.zarr"]
    elif round == 2:
        all_files = ["/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/base.zarr", 
                    f"/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/{user_id}/{protocol_folder}/round1.zarr",
                    f"/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/{user_id}/{protocol_folder}/round2.zarr"]
        demo_files = ["/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/base.zarr"]
    else:
        raise ValueError("Round must be 1 or 2")
    combine_and_weight_sirius(all_files, f"/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/{user_id}/{protocol_folder}/combined_round{round}.zarr", demo_files)