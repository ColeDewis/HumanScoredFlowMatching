import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def get_self_cross_scores(df, rater):
    self_scores = df[df["Demonstrator"] == rater][rater]
    cross_scores = df[df["Demonstrator"] != rater][rater]
    return self_scores, cross_scores

if __name__ == "__main__":
    ratings_banana = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - CutBanana.csv")
    ratings_cubes = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - BottleCubes.csv")
    ratings_cupball = pd.read_csv("/home/coled/720/3D-Diffusion-Policy/plotting/720 Demos - CupBallGood.csv")

    raters = ["Cole", "Jasper", "Sergey"]
    tasks = {
        "Cut Banana": ratings_banana,
        "Pickup Bottle": ratings_cubes,
        "Ball in Mug": ratings_cupball
    }

    # Prepare long-form DataFrame for seaborn
    plot_data = []
    for task_name, df in tasks.items():
        for rater in raters:
            self_scores, cross_scores = get_self_cross_scores(df, rater)
            for val in self_scores.dropna():
                plot_data.append({
                    "Task": task_name,
                    "Demonstrator": rater,
                    "Type": "Rating Self",
                    "Rating": val
                })
            for val in cross_scores.dropna():
                plot_data.append({
                    "Task": task_name,
                    "Demonstrator": rater,
                    "Type": "Rating Others",
                    "Rating": val
                })
        # Add overall average as a "Mean" demonstrator, with both Rating Self and Rating Others types
        all_ratings = pd.concat([df[r] for r in raters], ignore_index=True)
        for val in all_ratings.dropna():
            plot_data.append({
                "Task": task_name,
                "Demonstrator": "Task Mean",
                "Type": "Rating Self",
                "Rating": val
            })
            plot_data.append({
                "Task": task_name,
                "Demonstrator": "Task Mean",
                "Type": "Rating Others",
                "Rating": val
            })

    plot_df = pd.DataFrame(plot_data)

    palette = {"Rating Self": "#4C72B0", "Rating Others": "#C44E52"}

    # Plot each task as a separate figure
    for task_name in tasks.keys():
        plt.figure(figsize=(8, 6))
        ax = sns.violinplot(
            data=plot_df[plot_df["Task"] == task_name],
            x="Demonstrator",
            y="Rating",
            hue="Type",
            split=True,
            inner="quartile",
            palette=palette
        )
        plt.title(task_name, fontsize=20)
        plt.ylim(0.5, 10.5)
        plt.xlabel("Demonstrator", fontsize=18)
        plt.ylabel("Rating", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Make the "Mean" violin gray
        for violin in ax.collections:
            paths = violin.get_paths()
            if not paths:
                continue
            x = paths[0].vertices[:, 0].mean()
            if x > len(raters) - 0.5:
                violin.set_facecolor("#888888")
                violin.set_edgecolor("#444444")
                violin.set_alpha(1.0)

        # Add marker for the mean
        demo_order = raters + ["Task Mean"]
        star_handles = []
        for i, demo in enumerate(demo_order):
            for j, typ in enumerate(["Rating Self", "Rating Others"]):
                mean_val = plot_df[
                    (plot_df["Task"] == task_name) &
                    (plot_df["Demonstrator"] == demo) &
                    (plot_df["Type"] == typ)
                ]["Rating"].mean()
                if pd.notna(mean_val):
                    offset = -0.15 if typ == "Rating Self" else 0.15
                    # Gold star marker
                    h = ax.plot(i + offset, mean_val, marker="*", color="#FFD700", markersize=18, markeredgecolor="black", zorder=10)
                    if i == 0 and j == 0:
                        star_handles.append(h[0])

        # Custom legend with stars and violin colors, outside the plot
        handles, labels = ax.get_legend_handles_labels()
        custom_handles = [
            Line2D([0], [0], color="#4C72B0", lw=10, label="Rating Self"),
            Line2D([0], [0], color="#C44E52", lw=10, label="Rating Others"),
            Line2D([0], [0], marker="*", color="#FFD700", markeredgecolor="black", markersize=18, lw=0, label="Mean", linestyle="None")
        ]
        ax.legend(handles=custom_handles, fontsize=14, title="Type", title_fontsize=15, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        plt.tight_layout()
        plt.show()