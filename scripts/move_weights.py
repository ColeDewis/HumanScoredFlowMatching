import os
import shutil

if __name__ == "__main__":
    # weight_path = "/home/coled/720/3D-Diffusion-Policy/scripts/CutBanana"
    # target_path = "/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cut_banana"

    # weight_path = "/home/coled/720/3D-Diffusion-Policy/scripts/BottleCubes"
    # target_path = "/home/coled/720/3D-Diffusion-Policy/flow_policy/data/bottle_cubes"

    weight_path = "/home/coled/720/3D-Diffusion-Policy/scripts/CupBallGood"
    target_path = "/home/coled/720/3D-Diffusion-Policy/flow_policy/data/cup_ball"

    for idx in range(75):
        src_file = os.path.join(weight_path, f"{idx}.npy")
        dest_folder = os.path.join(target_path, str(idx))
        dest_file = os.path.join(dest_folder, f"weights_{idx}.npy")
        if os.path.isfile(src_file) and os.path.isdir(dest_folder):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} -> {dest_file}")
        else:
            print(f"Missing: {src_file} or {dest_folder}")