import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go


def provide_report(folder: str, report_json: str):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load JSON data into a DataFrame
    file_path = "results/" + folder + "/" + report_json + ".json"
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert the nested dictionary to a DataFrame
    df_list = []
    for cav, trusts in data.items():
        for other_cav, scores in trusts.items():
            for index, score in enumerate(scores):
                df_list.append(
                    {
                        "CAV": cav,
                        "Other_CAV": other_cav,
                        "Image_Index": index + 1,
                        "Trust_Score": score,
                    }
                )

    df = pd.DataFrame(df_list)
    df.to_csv(
        f"reports/{folder}/{folder}_detailed_trust_scores_{current_datetime}.csv",
        index=False,
    )

    # Calculate average trust score for each CAV pair
    average_scores = df.groupby(["CAV", "Other_CAV"]).mean()
    print(average_scores)

    # Saving the average scores to an Excel file
    average_scores.to_csv(
        f"reports/{folder}/{folder}_average_trust_scores_{current_datetime}.csv"
    )

    # Generate a pivot table
    pivot_table = df.pivot_table(
        values="Trust_Score", index="CAV", columns="Other_CAV", aggfunc="mean"
    )
    print(pivot_table)


def provide_visualize(folder: str, report_json: str):
    # Load JSON data into a DataFrame
    file_path = "results/" + folder + "/" + report_json + ".json"
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert the nested dictionary to a DataFrame
    df_list = []
    for cav, trusts in data.items():
        for other_cav, scores in trusts.items():
            for index, score in enumerate(scores):
                df_list.append(
                    {
                        "CAV": cav,
                        "Other_CAV": other_cav,
                        "Image_Index": index + 1,
                        "Trust_Score": score,
                    }
                )

    df = pd.DataFrame(df_list)

    # Create a single figure to hold all plots
    fig = go.Figure()

    # Add a trace for each unique pair of CAVs
    for _, row in df[["CAV", "Other_CAV"]].drop_duplicates().iterrows():
        cav = row["CAV"]
        other_cav = row["Other_CAV"]

        # Filter data for the specific pair
        filtered_df = df[(df["CAV"] == cav) & (df["Other_CAV"] == other_cav)]

        # Add a line trace for the filtered data
        fig.add_trace(
            go.Scatter(
                x=filtered_df["Image_Index"],
                y=filtered_df["Trust_Score"],
                mode="lines+markers",
                name=f"Trust from {cav} to {other_cav}",
            )
        )

    # Set chart titles and labels
    fig.update_layout(
        title="Trust Evolution Across All CAV Pairs",
        xaxis_title="Image Index",
        yaxis_title="Trust Score",
        template="plotly_dark",
    )

    # Show the plot
    fig.show()


def main():  #
    # Replace the folder and report to get the report and visualization accordingly
    # provide_report(folder="Sample0", report_json="Sample0_2024-06-15_14-56-40")
    provide_visualize(folder="Sample1", report_json="Sample1_2024-08-03_16-49-40")


if __name__ == "__main__":
    main()
