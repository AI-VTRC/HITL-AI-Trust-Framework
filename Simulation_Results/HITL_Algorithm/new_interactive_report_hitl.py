import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os

# Path to the json files
json_path = "json"


def get_json_files():
    """Function to get all json files from the json folder"""
    json_files = [f for f in os.listdir(json_path) if f.endswith(".json")]
    return json_files


def load_json_data(json_file):
    """Function to load JSON data based on selected file path"""
    file_path = os.path.join(json_path, json_file)
    with open(file_path) as f:
        return json.load(f)


def generate_traces_for_range(cav_data, selected_index):
    """Function to generate Plotly traces for a range of data points"""
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
        # Dropdown for selecting the json file
        dcc.Dropdown(
            id="json-dropdown",
            options=[
                {"label": json_file, "value": json_file}
                for json_file in get_json_files()
            ],
            placeholder="Select JSON File",
            value=None,
        ),
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
            max=0,
            step=1,
            value=0,
            marks={},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ]
)


# Callback to load the data and update the graph and slider
@app.callback(
    [
        Output("cav-interactions", "figure"),
        Output("data-point-slider", "max"),
        Output("data-point-slider", "marks"),
    ],
    [Input("json-dropdown", "value"), Input("data-point-slider", "value")],
)
def update_graph(selected_json, selected_index):
    if selected_json is None:
        return {}, 0, {}

    # Load the selected JSON data
    data = load_json_data(selected_json)

    # Generate traces for the selected index
    traces = generate_traces_for_range(data, selected_index)

    # Determine the max range of the slider
    max_data_points = max(
        len(values) for connections in data.values() for values in connections.values()
    )

    # Generate slider marks dynamically
    marks = {i: f"{i+1}" for i in range(max_data_points)}

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


# Callback to reset the dropdown and slider when the reset button is clicked
@app.callback(
    [Output("json-dropdown", "value"), Output("data-point-slider", "value")],
    [Input("reset-button", "n_clicks")],
    [State("json-dropdown", "value"), State("data-point-slider", "value")],
)
def reset_fields(n_clicks, json_value, slider_value):
    if n_clicks > 0:
        # Reset fields to their initial state
        return None, 0
    return json_value, slider_value


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
