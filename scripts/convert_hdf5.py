import os
import pickle
import re
import socket
import time

import h5py
import numpy as np
import torch
import torchvision
import tqdm
import zarr
from termcolor import cprint


def preproces_image(image):
    img_size = 84

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image


if __name__ == "__main__":
    # expert_data_path = "/home/coled/HumanScoredFlowMatching/flow_policy/data/banana_wam"
    # save_data_path = "/home/coled/HumanScoredFlowMatching/flow_policy/data/banana_wam_unwrapped.zarr"
    expert_data_path = "/home/coled/720/3D-Diffusion-Policy/flow_policy/data/franka_test"
    save_data_path = "/home/coled/720/3D-Diffusion-Policy/flow_policy/data/franka_test.zarr"

    dirs = os.listdir(expert_data_path)
    dirs = sorted(
        [f for f in dirs if re.match(r"episode_\d+\.h5", f)],
        key=lambda x: int(re.search(r"episode_(\d+)\.h5", x).group(1))
    )

    # storage
    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    if os.path.exists(save_data_path):
        cprint("Data already exists at {}".format(save_data_path), "red")
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = "y"
        if user_input == "y":
            cprint("Overwriting {}".format(save_data_path), "red")
            os.system("rm -rf {}".format(save_data_path))
        else:
            cprint("Exiting", "red")
            exit()
    os.makedirs(save_data_path, exist_ok=True)
    
    # check the first episode to see if we have images and point clouds
    # TODO this is brittle, and doesn't handle multiple images/pclouds.
    # The rest of the code also doesn't do that yet, but in future I will need
    # to handle that.
    data = h5py.File(os.path.join(expert_data_path, dirs[0]), "r")
    data = data["observations"]
    has_images = "image" in data.keys()
    has_pointclouds = "pointcloud" in data.keys()

    for i, demo_dir in enumerate(dirs):
        dir_name = os.path.dirname(demo_dir)

        cprint("Processing {}".format(demo_dir), "green")
        
        data = h5py.File(os.path.join(expert_data_path, demo_dir), "r")
        data = data["observations"]
        
        if has_images:
            demo_images = data["image"]
            img_arrays.extend(demo_images)
        if has_pointclouds:
            demo_pointclouds = data["pointcloud"]
            if i == 4: # HACK idk how this is mismatched???
                demo_pointclouds = demo_pointclouds[:-1]
            point_cloud_arrays.extend(demo_pointclouds)
        
        demo_cart = data["cartesian"]
        demo_joints = data["joints"]
        
        action = demo_cart["position"]
        robot_state = demo_joints["position"]
        gripper = data['gripper']
        gripper = np.where(gripper[:] < 0.01, 1, 0)
        

        
        # NOTE: we maybe should figure out a better angle rep rather than whatever this is..
        rotations = action[:, 3:6]
        unwrapped_rotations = np.unwrap(rotations, axis=0)
        action = np.concatenate([action[:, :3], unwrapped_rotations, gripper], axis=-1)

        action_arrays.extend(action)
        state_arrays.extend(robot_state)
        
        episode_len = robot_state.shape[0]
        
        episode_ends_arrays.append(total_count + episode_len)
        total_count += episode_len        

    # create zarr file
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    if has_images:
        img_arrays = np.stack(img_arrays, axis=0)
        if img_arrays.shape[1] == 3:  # make channel last
            img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
            
    if has_pointclouds:
        point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    
    action_arrays = np.stack(action_arrays, axis=0)
    state_arrays = np.stack(state_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    if has_images:
        img_chunk_size = (
            100,
            img_arrays.shape[1],
            img_arrays.shape[2],
            img_arrays.shape[3],
        )
        
    if has_pointclouds:
        point_cloud_chunk_size = (
            100,
            point_cloud_arrays.shape[1],
            point_cloud_arrays.shape[2],
        )
    # depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    if len(action_arrays.shape) == 2:
        action_chunk_size = (100, action_arrays.shape[1])
    elif len(action_arrays.shape) == 3:
        action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
    else:
        raise NotImplementedError
    
    if has_images:
        zarr_data.create_dataset(
            "img",
            data=img_arrays,
            chunks=img_chunk_size,
            dtype="uint8",
            overwrite=True,
            compressor=compressor,
        )
    
    if has_pointclouds:
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            dtype="float64",
            overwrite=True,
            compressor=compressor,
        )
        
    # zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(100,),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


    # print shape
    if has_images:
        cprint(
            f"img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]",
            "green",
        )
        
    if has_pointclouds:
        cprint(
            f"point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]",
            "green",
        )
    # cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(
        f"action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]",
        "green",
    )
    cprint(
        f"state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]",
        "green",
    )
    cprint(
        f"episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]",
        "green",
    )
    cprint(f"total_count: {total_count}", "green")
    cprint(f"Saved zarr file to {save_data_path}", "green")
