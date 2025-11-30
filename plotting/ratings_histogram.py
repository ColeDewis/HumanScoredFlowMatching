import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib.use('TkAgg')  # Use a non-interactive backend

def get_self_cross_scores(df, rater):
    self_scores = df[df["Demonstrator"] == rater][rater]
    cross_scores = df[df["Demonstrator"] != rater][rater]
    return self_scores, cross_scores

if __name__ == "__main__":
    ratings_banana = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - CutBanana.csv")
    ratings_cubes = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - BottleCubes.csv")
    ratings_cupball = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - CupBallGood.csv")

    raters = ["Cole", "Jasper", "Sergey"]

    bins = np.arange(1, 12)
    bin_centers = np.arange(1, 11)
    colors = ['#4C72B0', '#55A868', '#C44E52']

    # --- ALL RATINGS ---
    banana_scores = pd.concat([ratings_banana[r] for r in raters], ignore_index=True)
    cubes_scores = pd.concat([ratings_cubes[r] for r in raters], ignore_index=True)
    cupball_scores = pd.concat([ratings_cupball[r] for r in raters], ignore_index=True)

    counts_banana, _ = np.histogram(banana_scores.dropna(), bins=bins)
    counts_cubes, _ = np.histogram(cubes_scores.dropna(), bins=bins)
    counts_cupball, _ = np.histogram(cupball_scores.dropna(), bins=bins)
    y_max = max(counts_banana.max(), counts_cubes.max(), counts_cupball.max())

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    axs[0].bar(bin_centers, counts_banana, width=0.8, color=colors[0], edgecolor='black')
    axs[1].bar(bin_centers, counts_cubes, width=0.8, color=colors[1], edgecolor='black')
    axs[2].bar(bin_centers, counts_cupball, width=0.8, color=colors[2], edgecolor='black')
    for ax, title in zip(axs, ['Cut Banana', 'Pickup Bottle', 'Ball in Mug']):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_centers, fontsize=13)
        ax.set_ylim(0, y_max + 1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='x', labelbottom=True)
        # ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlabel('Rating', fontsize=14)
    plt.subplots_adjust(hspace=0.32)
    plt.tight_layout()

    # --- SELF RATINGS ---
    banana_self = pd.concat([get_self_cross_scores(ratings_banana, r)[0] for r in raters], ignore_index=True)
    cubes_self = pd.concat([get_self_cross_scores(ratings_cubes, r)[0] for r in raters], ignore_index=True)
    cupball_self = pd.concat([get_self_cross_scores(ratings_cupball, r)[0] for r in raters], ignore_index=True)

    counts_banana_self, _ = np.histogram(banana_self.dropna(), bins=bins)
    counts_cubes_self, _ = np.histogram(cubes_self.dropna(), bins=bins)
    counts_cupball_self, _ = np.histogram(cupball_self.dropna(), bins=bins)
    y_max_self = max(counts_banana_self.max(), counts_cubes_self.max(), counts_cupball_self.max())

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    axs[0].bar(bin_centers, counts_banana_self, width=0.8, color=colors[0], edgecolor='black')
    axs[1].bar(bin_centers, counts_cubes_self, width=0.8, color=colors[1], edgecolor='black')
    axs[2].bar(bin_centers, counts_cupball_self, width=0.8, color=colors[2], edgecolor='black')
    for ax, title in zip(axs, ['CutBanana', 'BottleCubes', 'CupBall']):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_centers, fontsize=13)
        ax.set_ylim(0, y_max_self + 1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlabel('Rating', fontsize=14)
    plt.subplots_adjust(hspace=0.32)

    # --- CROSS RATINGS ---
    banana_cross = pd.concat([get_self_cross_scores(ratings_banana, r)[1] for r in raters], ignore_index=True)
    cubes_cross = pd.concat([get_self_cross_scores(ratings_cubes, r)[1] for r in raters], ignore_index=True)
    cupball_cross = pd.concat([get_self_cross_scores(ratings_cupball, r)[1] for r in raters], ignore_index=True)

    counts_banana_cross, _ = np.histogram(banana_cross.dropna(), bins=bins)
    counts_cubes_cross, _ = np.histogram(cubes_cross.dropna(), bins=bins)
    counts_cupball_cross, _ = np.histogram(cupball_cross.dropna(), bins=bins)
    y_max_cross = max(counts_banana_cross.max(), counts_cubes_cross.max(), counts_cupball_cross.max())

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    axs[0].bar(bin_centers, counts_banana_cross, width=0.8, color=colors[0], edgecolor='black')
    axs[1].bar(bin_centers, counts_cubes_cross, width=0.8, color=colors[1], edgecolor='black')
    axs[2].bar(bin_centers, counts_cupball_cross, width=0.8, color=colors[2], edgecolor='black')
    for ax, title in zip(axs, ['CutBanana', 'BottleCubes', 'CupBall']):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_centers, fontsize=13)
        ax.set_ylim(0, y_max_cross + 1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlabel('Rating', fontsize=14)
    plt.subplots_adjust(hspace=0.32)

    plt.show()