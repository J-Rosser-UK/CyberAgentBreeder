import sys

sys.path.append("src")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from base import initialize_session, Scaffold, Population
import random
from collections import OrderedDict
from rich import print


def compute_pareto_frontier(df, maximize_x=True, maximize_y=True):
    """
    Computes the Pareto frontier for a given DataFrame.

    Parameters:
    - df: pandas DataFrame with 'x' and 'y' columns.
    - maximize_x: Boolean, whether to maximize 'x'.
    - maximize_y: Boolean, whether to maximize 'y'.

    Returns:
    - pandas DataFrame containing the Pareto frontier.
    """
    # Sort the data based on 'x' and 'y'
    df_sorted = df.sort_values(
        by=["x", "y"], ascending=(not maximize_x, not maximize_y)
    ).reset_index(drop=True)

    # Initialize the Pareto frontier
    pareto_front = []
    current_max = -np.inf if maximize_y else np.inf

    for index, row in df_sorted.iterrows():
        y = row["y"]

        if (maximize_y and y > current_max) or (not maximize_y and y < current_max):
            pareto_front.append(row)
            current_max = y

    return pd.DataFrame(pareto_front)


def plot_pareto_frontiers(scaffolds, ax):
    """
    Plots Pareto frontiers (median safety vs median capability) and all points for a single population.
    Additionally, plots marker lines for Pareto best points in the first generation and others,
    and fills the area under the first Pareto frontier with 50% opacity, including the origin (0,0).
    """
    # Step 1: Extract Relevant Data from Scaffolds
    data = []
    for scaffold in scaffolds:
        if (
            not scaffold.scaffold_capability_ci_median
            or scaffold.scaffold_capability_ci_median == 0
        ):
            continue  # Skip scaffolds with median capability of 0
        data.append(
            {
                "generation_timestamp": scaffold.generation_timestamp,
                "median_capability": scaffold.scaffold_capability_ci_median,
                "median_safety": scaffold.scaffold_safety_ci_median,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    # Clean the DataFrame by removing rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["median_capability", "median_safety"]
    )

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No valid data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    # Step 2: Group by Generation
    generations = (
        df.groupby("generation_timestamp")
        .apply(lambda x: x.sort_values("median_capability"))
        .reset_index(drop=True)
    )

    # Determine the first generation based on sorted timestamps
    sorted_generations = df["generation_timestamp"].sort_values().unique()
    if len(sorted_generations) == 0:
        ax.text(
            0.5,
            0.5,
            "No generations available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    first_generation = sorted_generations[0]
    other_generations = sorted_generations[1:]

    # Define a colorblind-friendly colormap with 11 discrete colors
    cmap = plt.get_cmap(
        "spring", len(sorted_generations)
    )  # You can change "Greens" to any other colormap
    colors = cmap(np.linspace(0, 1, len(sorted_generations)))  # Generate 11 colors

    # Extract the first and last colors for filling
    fill_color_first = cmap(0.0)  # First color of the colormap
    fill_color_last = cmap(1.0)  # Last color of the colormap

    # Calculate jitter magnitude as 1% of the data range for both x and y
    capability_range = df["median_capability"].max() - df["median_capability"].min()
    safety_range = df["median_safety"].max() - df["median_safety"].min()

    jitter_x = (
        0.01 * capability_range if capability_range > 0 else 0.001
    )  # Set a small default if range is zero
    jitter_y = (
        0.01 * safety_range if safety_range > 0 else 0.001
    )  # Set a small default if range is zero

    # Ensure jitter values are finite
    if not np.isfinite(jitter_x):
        jitter_x = 0.001  # Default small jitter
    if not np.isfinite(jitter_y):
        jitter_y = 0.001  # Default small jitter

    # Step 3: Compute and Plot Pareto Frontiers and Scatter Points for Each Generation
    for i, gen in enumerate(sorted_generations):
        color = colors[i % len(colors)]
        gen_data = df[df["generation_timestamp"] == gen].rename(
            columns={"median_capability": "x", "median_safety": "y"}
        )

        # Apply jitter to x and y
        jitter_low_x = -jitter_x if jitter_x > 0 else 0
        jitter_high_x = jitter_x if jitter_x > 0 else 0
        jitter_low_y = -jitter_y if jitter_y > 0 else 0
        jitter_high_y = jitter_y if jitter_y > 0 else 0

        if jitter_high_x > 0 and jitter_high_y > 0:
            jittered_x = gen_data["x"] + np.random.uniform(
                jitter_low_x, jitter_high_x, size=gen_data.shape[0]
            )
            jittered_y = gen_data["y"] + np.random.uniform(
                jitter_low_y, jitter_high_y, size=gen_data.shape[0]
            )
        else:
            jittered_x = gen_data["x"]
            jittered_y = gen_data["y"]

        print(len(sorted_generations))
        # Plot scatter points with higher zorder
        ax.scatter(
            jittered_x,
            jittered_y,
            color=color,
            alpha=1,
            edgecolor="k",
            s=50,
            label=(
                f"Gen {i}" if i == 0 or i == len(sorted_generations) - 1 else ""
            ),  # Label only first two for legend clarity
            zorder=5,  # Increased zorder to ensure scatter is on top
        )

        # Compute Pareto frontier using original (non-jittered) data
        pareto_df = compute_pareto_frontier(gen_data, maximize_x=True, maximize_y=True)

        if pareto_df.empty:
            continue  # Skip plotting Pareto if no points

        # Sort Pareto frontier by 'x' to ensure lines are drawn correctly
        pareto_df = pareto_df.sort_values(by="x")

        # Plot Pareto frontier as straight lines between points with higher zorder
        ax.plot(
            pareto_df["x"],
            pareto_df["y"],
            marker="o",
            linestyle="-",
            color=color,
            linewidth=2,
            alpha=1,
            zorder=4,  # Ensure lines are above fills but below scatter points
        )

    # Step 4: Identify and Plot Pareto Best Points
    # Baseline Fill (First Generation)
    first_gen_data = df[df["generation_timestamp"] == first_generation].rename(
        columns={"median_capability": "x", "median_safety": "y"}
    )
    print(first_gen_data)
    pareto_first_gen = compute_pareto_frontier(
        first_gen_data, maximize_x=True, maximize_y=True
    )
    print("pareto_first_gen", pareto_first_gen)
    if not pareto_first_gen.empty:
        # Sort Pareto frontier by 'x' to ensure proper polygon creation
        pareto_first_gen_sorted = pareto_first_gen.sort_values(by="x")

        # Extract x and y values
        x_pareto = pareto_first_gen_sorted["x"].tolist()
        y_pareto = pareto_first_gen_sorted["y"].tolist()

        # Identify the maximum y value in Pareto frontier
        y_max = y_pareto[0]  # Assuming sorted for maximize_y

        # Define the polygon coordinates including the origin (0,0)
        polygon_x = [0, 0] + x_pareto + [x_pareto[-1], 0]
        polygon_y = [0, y_max] + y_pareto + [0, 0]

        # Fill the polygon with 50% opacity and lower zorder using the first color
        ax.fill(
            polygon_x,
            polygon_y,
            color=fill_color_first,  # Use first color from colormap
            alpha=0.3,
            label="Baseline Best Fill",  # Optional: Label for legend
            zorder=2,  # Lower zorder to be behind evolved fill
        )

    # Evolved Fill (Other Generations)
    if len(other_generations) > 0:
        other_gens_data = df[df["generation_timestamp"].isin(other_generations)].rename(
            columns={"median_capability": "x", "median_safety": "y"}
        )
        pareto_other_gens = compute_pareto_frontier(
            other_gens_data, maximize_x=True, maximize_y=True
        )
        print("pareto_other_gen", pareto_other_gens)
        if not pareto_other_gens.empty:
            # Sort Pareto frontier by 'x' to ensure proper polygon creation
            pareto_other_gens_sorted = pareto_other_gens.sort_values(by="x")

            # Extract x and y values
            x_pareto = pareto_other_gens_sorted["x"].tolist()
            y_pareto = pareto_other_gens_sorted["y"].tolist()

            # Identify the maximum y value in Pareto frontier
            y_max = y_pareto[0]  # Assuming sorted for maximize_y

            # Define the polygon coordinates including the origin (0,0)
            polygon_x = [0, 0] + x_pareto + [x_pareto[-1], 0]
            polygon_y = [0, y_max] + y_pareto + [0, 0]

            # Fill the polygon with 30% opacity and higher zorder using the last color
            ax.fill(
                polygon_x,
                polygon_y,
                color=fill_color_last,  # Use last color from colormap
                alpha=0.3,
                label="Evolved Best Fill",  # Optional: Label for legend
                zorder=1,  # Higher zorder to be in front of baseline fill
            )

    # Step 5: Set Dynamic Axis Limits
    # Determine the maximum x and y values in the data
    max_x = df["median_capability"].max()
    max_y = df["median_safety"].max()

    # Define a margin (e.g., 5% of the maximum value)
    margin = 0.05

    # Calculate dynamic upper limits with margin, capped at 1
    upper_x = min(max_x + margin * max_x, 1) if max_x > 0 else 1
    upper_y = min(max_y + margin * max_y, 1) if max_y > 0 else 1

    # Set the axis limits
    ax.set_xlim(0, upper_x)
    ax.set_ylim(0, upper_y)

    # Customize Subplot
    ax.set_xlabel("Median Capability", fontsize=10)
    ax.set_ylabel("Median Safety", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7, zorder=0)

    # Add Legend
    handles, labels = ax.get_legend_handles_labels()
    # To avoid duplicate labels
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")


def main():
    random.seed(42)
    population_id = "d44a351c-d454-4d2c-ae74-f7e0e88b9ce8"
    population_id = "dd43526d-9a36-41c3-89bb-2f71c7738040"

    populations = []
    scaffolds_per_population = []

    for session in initialize_session():

        try:

            latest_populations = (
                session.query(Population)
                .order_by(Population.population_timestamp.desc())
                .all()
            )

            if not latest_populations:
                print("No populations found.")
                return

            for population in latest_populations:

                scaffolds = (
                    session.query(Scaffold)
                    .filter_by(population_id=population.population_id)
                    .all()
                )
                if len(scaffolds) < 10:
                    continue
                populations.append(population)

                scaffolds_per_population.append(scaffolds)
        except StopIteration:
            print("Session generator exhausted.")
            return
        except Exception as e:
            print(f"An error occurred while fetching populations: {e}")
            return

        print(len(populations))

        # Set up 2x4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()  # Flatten to 1D array for easy iteration

        for idx, (population, scaffolds) in enumerate(
            zip(populations, scaffolds_per_population)
        ):
            ax = axes[idx]
            plot_pareto_frontiers(scaffolds, ax)
            ax.set_title(
                f"{population.population_benchmark}-{population.population_id}",
                fontsize=12,
            )

        # Remove any unused subplots if less than 8
        if len(populations) < len(axes):
            for idx in range(len(populations), len(axes)):
                fig.delaxes(axes[idx])

        for p in populations:
            print(
                p.population_benchmark, p.population_id, len(scaffolds_per_population)
            )

        plt.suptitle(
            "CyberAgentBreeder: Capability vs Safety Pareto Frontiers", fontsize=16
        )
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to accommodate suptitle
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Increase space between subplots
        plt.show()


if __name__ == "__main__":
    main()
