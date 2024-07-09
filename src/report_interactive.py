import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import json
import os

# Load the trust data
# folder = "Sample6"
# report_json = "Sample6_2024-06-16_18-00-12"

folder = "Sample5"
report_json = "Sample5_2024-06-16_16-59-23"

file_path = f"results/{folder}/{report_json}.json"
with open(file_path, "r") as file:
    trust_data = json.load(file)

# Convert the nested dictionary to a DataFrame
df_list = []
for cav, trusts in trust_data.items():
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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Function to get image paths
def get_image_paths(root_connection, num_cars, num_images):
    image_paths = [
        [
            os.path.join(root_connection, f"Car{i+1}", f"frame_{j+1}.jpg")
            for j in range(num_images)
        ]
        for i in range(num_cars)
    ]
    return image_paths


# Get image paths for the scroller
root_connection = "../data/" + folder
num_cars = 4
num_images = len(trust_data["cav1"]["cav2"])
image_paths = get_image_paths(root_connection, num_cars, num_images)

# Layout of the Dash app
app.layout = dbc.Container(
    [
        dcc.Store(id="legend-state", data={}),
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
                            dbc.CardImg(
                                id=f"car{i+1}-image", top=True, style={"width": "100%"}
                            ),
                            dbc.CardBody(
                                html.P(
                                    f"Car {i+1} - Frame {{image_index}}",
                                    className="text-center",
                                ),
                                id=f"car{i+1}-info",
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
    ],
    fluid=True,
)


# Define outputs for both images and their labels
output_list = []
for i in range(num_cars):
    output_list.append(Output(f"car{i+1}-image", "src"))
    output_list.append(Output(f"car{i+1}-info", "children"))


@app.callback(output_list, [Input("image-slider", "value")])
def update_images_and_labels(image_index):
    updates = []
    for i in range(num_cars):
        image_src = f"/assets/data/{folder}/Car{i+1}/frame_{image_index}.jpg"
        label = f"Car {i+1} - Frame {image_index}"
        updates.extend([image_src, label])
    return updates


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
