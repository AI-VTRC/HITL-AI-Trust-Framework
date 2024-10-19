import streamlit as st
import plotly.graph_objs as go
import os
import json
from PIL import Image

# Set the page to wide mode (full screen)
st.set_page_config(layout="wide", page_title="CAVs Interaction Visualization")

# Define the base directories
base_dirs = {
    "HITL_Algorithm": "HITL_Algorithm/json",
    "Original_Algorithm": "Original_Algorithm/json",
}

# Define the base image directory
image_base_dir = "../data_bbxs"


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


def generate_traces_for_range(cav_data, selected_index):
    traces = []
    for cav, connections in cav_data.items():
        for target_cav, values in connections.items():
            if isinstance(values, (list, tuple)):
                # Slice the values for the selected index range
                x_values = list(range(1, selected_index + 2))
                y_values = values[: selected_index + 1]

                # Create the trace with text displaying the y-values
                traces.append(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers+text",  # Add text mode to display values
                        text=[
                            f"{y:.2f}" for y in y_values
                        ],  # Format the values to show
                        textposition="top center",  # Position the text above the markers
                        textfont=dict(color="black"),  # Set text color to black
                        name=f"{cav} -> {target_cav}",
                    )
                )
            else:
                st.warning(f"Unexpected data format for {cav} -> {target_cav}")
    return traces


# Function to get image paths for the current frame
def get_image_paths(sample_folder, frame_number):
    images = {}
    for i in range(1, 5):
        if sample_folder == "Sample1":
            image_file = f"Car{i}_frame_{frame_number}_with_boxes.jpg"
        else:
            image_file = f"Car{i}_frame_{frame_number-1}_with_boxes.jpg"
        image_path = os.path.join(image_base_dir, sample_folder, image_file)
        images[f"Car{i}"] = image_path if os.path.exists(image_path) else None
    return images


# Center the title using custom HTML and CSS
st.markdown(
    "<h1 style='text-align: center;'>CAVs Interaction Visualization</h1>",
    unsafe_allow_html=True,
)

# Dropdown for selecting the algorithm
algorithm = st.selectbox("Select Algorithm", ["HITL_Algorithm", "Original_Algorithm"])

if algorithm:
    # Dropdown for selecting the sample folder
    sample_folders = get_samples(algorithm)
    sample_folder = st.selectbox("Select Sample Folder", sample_folders)

    if sample_folder:
        # Dropdown for selecting the JSON file
        json_files = get_json_files(algorithm, sample_folder)
        json_file = st.selectbox("Select JSON File", json_files)

        if json_file:
            # Load the selected JSON data
            data = load_json_data(algorithm, sample_folder, json_file)

            # Slider for selecting the range of data points
            max_data_points = max(
                len(values)
                for connections in data.values()
                for values in connections.values()
            )
            selected_index = st.slider(
                "Select Data Point Index", 0, max_data_points - 1, 0
            )

            # Generate traces for the selected index and plot the graph
            traces = generate_traces_for_range(data, selected_index)
            fig = go.Figure(traces)
            fig.update_layout(
                title=dict(
                    text=f"CAV Interaction Data (Showing Data Points: 1 to {selected_index + 1})",
                    font=dict(color="black"),
                ),
                xaxis=dict(
                    title="Time Frame Index",
                    tickfont=dict(color="black"),
                    titlefont=dict(color="black", size=14, family="Arial Black"),
                ),
                yaxis=dict(
                    title="Trust Value",
                    tickfont=dict(color="black"),
                    titlefont=dict(color="black", size=14, family="Arial Black"),
                ),
                hovermode="closest",
                template="plotly_white",  # Use light theme for Plotly charts
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display images for 4 cars
            images = get_image_paths(sample_folder, selected_index + 1)
            cols = st.columns(4)
            for i in range(4):
                car_image = images.get(f"Car{i+1}")
                if car_image and os.path.exists(car_image):
                    with cols[i]:
                        st.image(Image.open(car_image), use_column_width=True)
                        st.markdown(
                            f"<h3 style='text-align:center; color:black;'>CAV{i+1} FOV</h3>",
                            unsafe_allow_html=True,
                        )
                else:
                    with cols[i]:
                        # Display the message in black using markdown
                        st.markdown(
                            f"<p style='text-align:center; color:black;'>Car {i+1} image not found</p>",
                            unsafe_allow_html=True,
                        )
