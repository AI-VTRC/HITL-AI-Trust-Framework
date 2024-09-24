import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import json
import os
from flask import send_from_directory
import re


# Function to get available folders, current_datetime, and report JSON files
def get_available_reports(results_base_dir="results"):
    folders = []
    data = {}

    # Traverse through directories to find the correct structure
    for folder in os.listdir(results_base_dir):
        # Match folders starting with a date like "8_29_24_scenario_1"
        if re.match(r"^\d{1,2}_\d{1,2}_\d{2}_scenario_\d+", folder):
            folder_path = os.path.join(results_base_dir, folder)
            if os.path.isdir(folder_path):
                folders.append(folder)
                data[folder] = []

                # Look inside the folder for the current_datetime subfolder
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path) and re.match(
                        r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", subfolder
                    ):
                        current_datetime = subfolder

                        # Look for JSON files in the current_datetime subfolder
                        for report_file in os.listdir(subfolder_path):
                            if report_file.endswith(".json"):
                                data[folder].append(
                                    {
                                        "current_datetime": current_datetime,
                                        "report_json": report_file,
                                    }
                                )

    return folders, data


folders, data = get_available_reports()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Function to get image paths dynamically based on selections
def get_image_paths(results_dir, num_cars, num_images):
    image_paths = [
        [
            os.path.join(results_dir, f"Car{i+1}_frame_{j+1}_with_boxes.jpg")
            for j in range(num_images)
        ]
        for i in range(num_cars)
    ]
    return image_paths


# Serve static images
@server.route("/results/<folder>/<datetime>/<path:filename>")
def serve_image(folder, datetime, filename):
    image_directory = f"results/{folder}/{datetime}"
    return send_from_directory(image_directory, filename)


# Add dropdowns for dynamic selection of folder, datetime, and report JSON
app.layout = dbc.Container(
    [
        dcc.Store(id="legend-state", data={}),
        dcc.Store(id="enlarged-image-src", data=""),
        dbc.Row(
            dbc.Col(
                html.H1("CAV Trust Evaluation Dashboard"),
                className="text-center",
            )
        ),
        # Dropdowns for folder, datetime, and report selection
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="folder-dropdown",
                        options=[
                            {"label": folder, "value": folder} for folder in folders
                        ],
                        value=folders[0],  # Default to the first folder
                        placeholder="Select Folder",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="datetime-dropdown",
                        placeholder="Select Datetime",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="report-dropdown",
                        placeholder="Select Report",
                    ),
                    width=4,
                ),
            ]
        ),
        dbc.Row(html.Div(style={"height": "40px"})),
        dbc.Row(
            dbc.Col(dcc.Graph(id="trust-plot"), width=12),
        ),
        dbc.Row(html.Div(style={"height": "40px"})),
        dbc.Row(
            dbc.Col(
                dcc.Slider(
                    id="image-slider",
                    min=1,
                    max=1,  # Default max will be updated dynamically
                    step=1,
                    value=1,
                    marks={1: "1"},  # Default marks
                ),
            )
        ),
        dbc.Row(html.Div(style={"height": "40px"})),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            html.Div(
                                dbc.CardImg(
                                    id=f"car{i+1}-image",
                                    top=True,
                                    style={"width": "100%", "cursor": "pointer"},
                                ),
                                id=f"car{i+1}-div",
                                n_clicks=0,
                            ),
                            dbc.CardBody(
                                html.Div(
                                    [
                                        html.P(
                                            f"Car {i+1} - Frame {{image_index}}",
                                            className="text-center",
                                            id=f"car{i+1}-info",
                                        ),
                                        html.Ul(id=f"car{i+1}-objects"),
                                    ]
                                ),
                            ),
                        ],
                        style={
                            "width": "18rem",
                            "margin": "10px",
                        },  # Adjust the width and margin as needed
                    ),
                    width=3,
                )
                for i in range(4)  # Assuming 4 cars based on original code
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                html.Img(
                    id="folder-image",
                    style={
                        "width": "20%",
                        "margin-top": "20px",
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                    },
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Img(
                    id="enlarged-image",
                    style={
                        "width": "60%",
                        "margin-top": "20px",
                        "display": "block",
                        "margin-left": "auto",
                        "margin-right": "auto",
                    },
                )
            )
        ),
    ],
    fluid=True,
)


# Callback to update the datetime and report dropdowns when a folder is selected
@app.callback(
    [
        Output("datetime-dropdown", "options"),
        Output("datetime-dropdown", "value"),
        Output("folder-image", "src"),
    ],  # Update folder image
    [Input("folder-dropdown", "value")],
)
def update_datetime_dropdown(selected_folder):
    if selected_folder:
        datetime_options = [
            {"label": entry["current_datetime"], "value": entry["current_datetime"]}
            for entry in data[selected_folder]
        ]
        folder_image_src = f"/assets/data/{selected_folder}/{selected_folder}.jpg"  # Update image source
        return (
            datetime_options,
            datetime_options[0]["value"] if datetime_options else None,
            folder_image_src,
        )
    return [], None, None


# Callback to update the report dropdown when a datetime is selected
@app.callback(
    [Output("report-dropdown", "options"), Output("report-dropdown", "value")],
    [Input("folder-dropdown", "value"), Input("datetime-dropdown", "value")],
)
def update_report_dropdown(selected_folder, selected_datetime):
    if selected_folder and selected_datetime:
        report_options = [
            {"label": entry["report_json"], "value": entry["report_json"]}
            for entry in data[selected_folder]
            if entry["current_datetime"] == selected_datetime
        ]
        return report_options, report_options[0]["value"] if report_options else None
    return [], None


# Callback to update the image slider and load the JSON data when the report is selected
@app.callback(
    [
        Output("image-slider", "max"),
        Output("image-slider", "marks"),
        Output("trust-plot", "figure"),
        *[Output(f"car{i+1}-image", "src") for i in range(4)],
    ],  # Update car images
    [
        Input("folder-dropdown", "value"),
        Input("datetime-dropdown", "value"),
        Input("report-dropdown", "value"),
        Input("image-slider", "value"),
    ],
)
def update_content(selected_folder, selected_datetime, selected_report, image_index):
    if not all([selected_folder, selected_datetime, selected_report]):
        return 1, {1: "1"}, go.Figure(), [""] * 4

    # Construct file path and load the JSON data
    file_path = f"results/{selected_folder}/{selected_datetime}/{selected_report}"
    with open(file_path, "r") as file:
        trust_data = json.load(file)

    # Process the JSON data into a DataFrame
    df_list = []
    for cav, data in trust_data.items():
        for other_cav, scores in data["trust_scores"].items():
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

    # Get the number of images and image paths
    num_images = len(trust_data["cav1"]["trust_scores"]["cav2"])
    slider_marks = {i: str(i) for i in range(1, num_images + 1)}

    results_dir = f"results/{selected_folder}/{selected_datetime}"
    image_paths = get_image_paths(results_dir, 4, num_images)  # Assuming 4 cars

    # Get the image sources for the current slider value
    image_sources = [
        f"/results/{selected_folder}/{selected_datetime}/Car{i+1}_frame_{image_index}_with_boxes.jpg"
        for i in range(4)
    ]

    # Create the trust plot
    fig = go.Figure()
    for _, row in df[["CAV", "Other_CAV"]].drop_duplicates().iterrows():
        cav = row["CAV"]
        other_cav = row["Other_CAV"]

        # Filter data for the specific pair
        filtered_df = df[
            (df["CAV"] == cav)
            & (df["Other_CAV"] == other_cav)
            & (df["Image_Index"] <= image_index)
        ]

        # Add a line trace for the filtered data
        fig.add_trace(
            go.Scatter(
                x=filtered_df["Image_Index"],
                y=filtered_df["Trust_Score"],
                mode="lines+markers",
                name=f"Trust from {cav} to {other_cav}",
            )
        )

    fig.update_layout(
        title="Trust Evolution Across All CAV Pairs",
        xaxis_title="Image (Frame) Index",
        yaxis_title="Trust Score",
        template="plotly_dark",
    )

    return num_images, slider_marks, fig, *image_sources


if __name__ == "__main__":
    app.run_server(debug=True)
