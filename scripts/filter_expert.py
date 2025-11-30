import os
import shutil

import numpy as np


def filter_and_copy(expert_data_path, scores_files_path, target_path):
    # Ensure the target directory exists
    os.makedirs(target_path, exist_ok=True)

    # Initialize logging
    log = []
    total_matches = 0

    # Iterate through all score files (0.npy to 74.npy)
    for i in range(75):  # Assuming files are named 0.npy to 74.npy
        score_file = os.path.join(scores_files_path, f"{i}.npy")
        if os.path.exists(score_file):
            # Load the score
            score = np.load(score_file)
            if score >= 8:
                # Log the match
                log.append(f"Match found: {score_file} with value {score}")
                total_matches += 1

                # Copy the corresponding folder
                folder_to_copy = os.path.join(expert_data_path, str(i))
                if os.path.exists(folder_to_copy):
                    shutil.copytree(folder_to_copy, os.path.join(target_path, str(i)))
                else:
                    log.append(f"Folder not found: {folder_to_copy}")
        else:
            log.append(f"Score file not found: {score_file}")

    # Log the total matches
    log.append(f"Total matches: {total_matches}")

    # Print the log
    for entry in log:
        print(entry)

if __name__ == '__main__':
    # expert_data_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/bottle_cubes'
    # scores_files_path = '/home/coled/720/3D-Diffusion-Policy/scripts/BottleCubes'
    # target_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/bottle_cubes_expert'
    
    # 52 for bottlecubes
    # 45 for cupball
    # 42 for cut banana
    
    # expert_data_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cup_ball'
    # scores_files_path = '/home/coled/720/3D-Diffusion-Policy/scripts/CupBallGood'
    # target_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cup_ball_expert'

    expert_data_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cut_banana'
    scores_files_path = '/home/coled/720/3D-Diffusion-Policy/scripts/CutBanana'
    target_path = '/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cut_banana_expert'
    filter_and_copy(expert_data_path, scores_files_path, target_path)