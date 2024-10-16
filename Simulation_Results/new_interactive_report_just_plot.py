import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os

st.set_page_config(layout="wide")

# Define the base directories
base_dirs = {
    "HITL_Algorithm": "HITL_Algorithm/json",
    "Original_Algorithm": "Original_Algorithm/json",
}


# Function to get available folders (Samples) from the selected base directory
def get_samples(base_dir):
    folder_path = base_dirs[base_dir]
    folders = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]
    )
    return folders


# Function to get JSON files from the selected sample folder
def get_json_files(base_dir, sample_folder):
    folder_path = os.path.join(base_dirs[base_dir], sample_folder)
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    return json_files


# Function to load JSON data based on selected file path
def load_json_data(base_dir, folder, json_file):
    file_path = os.path.join(base_dirs[base_dir], folder, json_file)
    with open(file_path) as f:
        return json.load(f)


# Function to generate Plotly traces for a range of data points
def generate_traces_for_range(cav_data, selected_index):
    traces = []
    for cav, connections in cav_data.items():
        for target_cav, values in connections.items():
            traces.append(
                go.Scatter(
                    x=list(range(1, selected_index + 2)),
                    y=values[: selected_index + 1],
                    mode="lines",
                    name=f"{cav} -> {target_cav}",
                )
            )
    return traces


# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("CAV Interaction Visualization", style={"textAlign": "center"}),
        # Dropdown for selecting the algorithm (HITL_Algorithm or Original_Algorithm)
        dcc.Dropdown(
            id="algorithm-dropdown",
            options=[
                {"label": "HITL_Algorithm", "value": "HITL_Algorithm"},
                {"label": "Original_Algorithm", "value": "Original_Algorithm"},
            ],
            value=None,
            placeholder="Select Algorithm",
        ),
        # Dropdown for selecting the sample (based on selected algorithm)
        dcc.Dropdown(
            id="sample-dropdown", placeholder="Select Sample Folder", value=None
        ),
        # Dropdown for selecting the json file (based on selected sample)
        dcc.Dropdown(id="json-dropdown", placeholder="Select JSON File", value=None),
        # Reset button
        html.Button(
            "Reset",
            id="reset-button",
            n_clicks=0,
            style={"marginTop": "10px", "marginBottom": "10px"},
        ),
        # Graph component
        dcc.Graph(id="cav-interactions"),
        # Slider for selecting the range of data points to display
        dcc.Slider(
            id="data-point-slider",
            min=0,
            max=0,  # This will be updated dynamically based on the JSON file
            step=1,
            value=0,
            marks={},  # Marks will be set dynamically
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ]
)


# Callback to update the sample folder dropdown based on the selected algorithm
@app.callback(
    Output("sample-dropdown", "options"), [Input("algorithm-dropdown", "value")]
)
def update_sample_dropdown(selected_algorithm):
    if selected_algorithm is None:
        return []
    samples = get_samples(selected_algorithm)
    return [{"label": sample, "value": sample} for sample in samples]


# Callback to update the json file dropdown based on the selected sample folder
@app.callback(
    Output("json-dropdown", "options"),
    [Input("sample-dropdown", "value"), Input("algorithm-dropdown", "value")],
)
def update_json_dropdown(selected_sample, selected_algorithm):
    if selected_sample is None or selected_algorithm is None:
        return []
    json_files = get_json_files(selected_algorithm, selected_sample)
    return [{"label": json_file, "value": json_file} for json_file in json_files]


# Callback to load the data and update the graph and slider
@app.callback(
    [
        Output("cav-interactions", "figure"),
        Output("data-point-slider", "max"),
        Output("data-point-slider", "marks"),
    ],
    [
        Input("algorithm-dropdown", "value"),
        Input("sample-dropdown", "value"),
        Input("json-dropdown", "value"),
        Input("data-point-slider", "value"),
    ],
)
def update_graph(selected_algorithm, selected_sample, selected_json, selected_index):
    if selected_algorithm is None or selected_sample is None or selected_json is None:
        return {}, 0, {}

    # Load the selected JSON data
    data = load_json_data(selected_algorithm, selected_sample, selected_json)

    # Generate traces for the selected index
    traces = generate_traces_for_range(data, selected_index)

    # Determine the max range of the slider
    max_data_points = max(
        len(values) for connections in data.values() for values in connections.values()
    )

    # Generate slider marks dynamically
    marks = {i: f"{i + 1}" for i in range(max_data_points)}

    return (
        {
            "data": traces,
            "layout": go.Layout(
                title=f"CAV Interaction Data (Showing Data Points: 1 to {selected_index + 1})",
                title_x=0.5,
                xaxis={"title": "Data Point Index"},
                yaxis={"title": "Value"},
                hovermode="closest",
            ),
        },
        max_data_points - 1,
        marks,
    )


# Callback to reset the dropdowns and slider when the reset button is clicked
@app.callback(
    [
        Output("algorithm-dropdown", "value"),
        Output("sample-dropdown", "value"),
        Output("json-dropdown", "value"),
        Output("data-point-slider", "value"),
    ],
    [Input("reset-button", "n_clicks")],
    [
        State("algorithm-dropdown", "value"),
        State("sample-dropdown", "value"),
        State("json-dropdown", "value"),
        State("data-point-slider", "value"),
    ],
)
def reset_fields(n_clicks, algorithm_value, sample_value, json_value, slider_value):
    if n_clicks > 0:
        # Reset all fields to None or initial values
        return None, None, None, 0
    return algorithm_value, sample_value, json_value, slider_value


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
