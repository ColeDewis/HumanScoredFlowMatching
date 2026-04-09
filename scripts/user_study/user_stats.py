import argparse
import os
import pickle
import re
import socket
import time
from itertools import groupby

import h5py
import numpy as np
import torch
import torchvision
import tqdm
import zarr
from termcolor import cprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a user's h5's to zarr for training.")
    
    users = ["ilya", "riley", "tanner", "olivia", "amir"]
    rounds = [1, 2]
    protocols = ["single", "multiple"]
    
    for protocol_folder in protocols:
        for round in rounds:
    
            cprint(f"Protocol: {protocol_folder}, Round: {round}", "green")
    
            intervention_steps_arr = []
            intervention_ratio_arr = []
            takeovers_per_rollout_arr = []
            avg_takeover_lengths_arr = []
            
            for user_id in users:

                h5s_dir = f"/home/coled/720/3D-Diffusion-Policy/flow_policy/data_sirius/user_study/{user_id}/{protocol_folder}/round{round}"

                dirs = os.listdir(h5s_dir)
                dirs = sorted(
                    [f for f in dirs if re.match(r"episode_\d+\.h5", f)],
                    key=lambda x: int(re.search(r"episode_(\d+)\.h5", x).group(1))
                )

                takeover_arrays = []

                data = h5py.File(os.path.join(h5s_dir, dirs[0]), "r")

                total_takeover_steps = 0
                total_steps = 0
                total_takeovers = 0
                indv_takeover_steps = []
                
                for i, demo_dir in enumerate(dirs):
                    dir_name = os.path.dirname(demo_dir)

                    # cprint("Processing {}".format(demo_dir), "green")
                    
                    data = h5py.File(os.path.join(h5s_dir, demo_dir), "r")
                    data = data["observations"]
                    
                    # print(data["takeover"].shape)
                    # print(data["takeover"][:, 0])
                    takeover_steps = np.sum(data["takeover"])
                    
                    total_takeover_steps += takeover_steps
                    total_steps += data["takeover"].shape[0]
                    # BELOW WRONG
                    takeover_lens = [sum(1 for _ in group) for value, group in groupby(data["takeover"][:, 0]) if value]
                    takeover_lens = [x for x in takeover_lens if x != 1]
                    indv_takeover_steps.extend(takeover_lens)
                    # print(data["takeover"][:, 0])
                    # print(takeover_lens)
                    # exit()
                    
                    takeover_starts = (data["takeover"][1:] == True) & (data["takeover"][:-1] == False)
                    takeover_count = np.sum(takeover_starts)
                    # print(takeover_count)
                    total_takeovers += takeover_count
                    
                # cprint(f"Data for user: {user_id}", "green")
                # # print(total_takeover_steps, total_steps, np.round(total_takeover_steps/total_steps, 3))
                # print(f"{np.round(total_takeover_steps/total_steps, 3) * 100}% of the data is an intervention.")
                # print(f"In total there was {total_takeover_steps} steps of takeover data added.")
                # # print(total_takeovers, total_takeovers / 10)
                # print(f"There was {total_takeovers / 10} takeovers per rollout")
                # print(f"Average takeover length was: {np.sum(indv_takeover_steps)/len(indv_takeover_steps)} steps")
                
                intervention_steps_arr.append(total_takeover_steps)
                intervention_ratio_arr.append(total_takeover_steps/total_steps)
                takeovers_per_rollout_arr.append(total_takeovers/10)
                avg_takeover_lengths_arr.append(np.sum(indv_takeover_steps)/len(indv_takeover_steps))
                
                
            cprint("AVERAGES: ", "cyan")
            print(f"Average total intervention steps: {np.average(intervention_steps_arr)}, {np.std(intervention_steps_arr)}")
            print(f"Average intervention pct: {np.average(intervention_ratio_arr)}, {np.std(intervention_ratio_arr)}")
            print(f"Average takeovers per rollout: {np.average(takeovers_per_rollout_arr)}, {np.std(takeovers_per_rollout_arr)}")
            print(f"Average takeover length: {np.average(avg_takeover_lengths_arr)}, {np.std(avg_takeover_lengths_arr)}")