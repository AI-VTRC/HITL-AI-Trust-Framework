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

# Load the trust data
# folder = "Sample0"
# report_json = "Sample0_2024-08-03_21-11-05_threshold_0.8"
folder = "8_29_24_scenario_1"
report_json = "8_29_24_scenario_1_2024-08-29_23-05-53_threshold_0.8"

# Assume the current_datetime is known or passed
# current_datetime = "2024-08-03_21-11-05"
current_datetime = "2024-08-29_23-05-53"
results_dir = f"results/{folder}/{current_datetime}"

file_path = os.path.join(results_dir, f"{folder}_{current_datetime}_threshold_0.8.json")
with open(file_path, "r") as file:
    trust_data = json.load(file)

# Convert the nested dictionary to a DataFrame
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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Serve static images
@server.route("/results/<folder>/<datetime>/<path:filename>")
def serve_image(folder, datetime, filename):
    image_directory = f"results/{folder}/{datetime}"
    return send_from_directory(image_directory, filename)


# Function to get image paths
def get_image_paths(results_dir, num_cars, num_images):
    image_paths = [
        [
            os.path.join(results_dir, f"Car{i+1}_frame_{j+1}_with_boxes.jpg")
            for j in range(num_images)
        ]
        for i in range(num_cars)
    ]
    return image_paths


# Get image paths for the scroller
num_cars = 4
num_images = len(trust_data["cav1"]["trust_scores"]["cav2"])
image_paths = get_image_paths(results_dir, num_cars, num_images)

# Layout of the Dash app
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
        dbc.Row(
            dbc.Col(
                html.H6(report_json),
                width={"size": 6, "offset": 3},
                className="text-center",
            )
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
                    max=num_images,
                    step=1,
                    value=1,
                    marks={i: str(i) for i in range(1, num_images + 1)},
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
                for i in range(num_cars)
            ],
            className="mb-4",
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
        dbc.Row(html.Div(style={"height": "40px"})),
        dbc.Row(
            dbc.Col(
                html.Img(
                    src=f'assets/data/{folder}/{folder}.jpg',
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
    ],
    fluid=True,
)

# Define outputs for both images and their labels
output_list = []
for i in range(num_cars):
    output_list.append(Output(f"car{i+1}-image", "src"))
    output_list.append(Output(f"car{i+1}-info", "children"))
    output_list.append(Output(f"car{i+1}-objects", "children"))


@app.callback(output_list, [Input("image-slider", "value")])
def update_images_and_labels(image_index):
    updates = []
    for i in range(num_cars):
        image_src = f"results/{folder}/{current_datetime}/Car{i+1}_frame_{image_index}_with_boxes.jpg"
        label = f"Car {i+1} - Frame {image_index}"
        objects = trust_data[f"cav{i+1}"]["detected_objects"][image_index - 1][
            "objects"
        ]
        object_list = [
            html.Li(f"{obj['label']}: {obj['confidence']:.2f}") for obj in objects
        ]
        updates.extend([image_src, label, object_list])
    return updates


# Callback to update the enlarged image when any car image is clicked
@app.callback(
    Output("enlarged-image", "src"),
    [Input(f"car{i+1}-div", "n_clicks") for i in range(num_cars)],
    [State("image-slider", "value")],
)
def enlarge_image(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    else:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        car_index = int(
            triggered_id.replace("car", "").replace("-div", "")
        )  # Extract car index from id
        slider_value = int(args[-1])
        image_src = f"/results/{folder}/{current_datetime}/Car{car_index}_frame_{slider_value}_with_boxes.jpg"
        return image_src


# Update the trust plot based on the slider value
@app.callback(
    Output("trust-plot", "figure"),
    [Input("image-slider", "value")],
)
def update_trust_plot(image_index):
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
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
