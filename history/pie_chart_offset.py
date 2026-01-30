import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import numpy as np

def plot_well_pie_charts(data):
    """
    Processes the well data and plots pie charts on a map.

    Args:
        data (pd.DataFrame): DataFrame with columns ['Well', 'X', 'Y', 'Layer', 'Thickness'].
    """
    if data.empty:
        st.warning("The uploaded data is empty.")
        return

    # --- 1. UI Controls and Data Setup ---
    st.sidebar.subheader("Display Options")
    show_well_names = st.sidebar.checkbox("Show Well Names", value=True)

    st.sidebar.subheader("Layer Colors")
    unique_layers = data['Layer'].unique()
    default_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    
    layer_color_map = {}
    for i, layer in enumerate(unique_layers):
        # Convert default RGBA to Hex for color picker
        default_hex = '#%02x%02x%02x' % (int(default_colors[i][0]*255), int(default_colors[i][1]*255), int(default_colors[i][2]*255))
        color = st.sidebar.color_picker(f"Color for {layer}", value=default_hex)
        layer_color_map[layer] = color

    # Group data by well to calculate total thickness
    well_data = data.groupby('Well').agg({
        'X': 'first',
        'Y': 'first',
        'Thickness': 'sum'
    }).rename(columns={'Thickness': 'TotalThickness'})

    # --- Add filter controls ---
    st.sidebar.subheader("Data Filters")
    if not well_data.empty:
        min_thick = float(well_data['TotalThickness'].min())
        max_thick = float(well_data['TotalThickness'].max())

        min_filter, max_filter = st.sidebar.slider(
            "Filter by Total Well Thickness",
            min_value=min_thick,
            max_value=max_thick,
            value=(min_thick, max_thick)
        )

        # Apply the filter
        well_data = well_data[(well_data['TotalThickness'] >= min_filter) & (well_data['TotalThickness'] <= max_filter)]

    # Get the max thickness from the filtered group to scale the radius of each pie
    max_filtered_thickness = well_data['TotalThickness'].max() if not well_data.empty else 1


    # --- 2. Plotting Setup ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Well Layer Thickness Pie Chart Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='box')

    # Set plot limits with a small margin
    x_range = data['X'].max() - data['X'].min()
    y_range = data['Y'].max() - data['Y'].min()
    ax.set_xlim(data["X"].min() - x_range * 0.1, data["X"].max() + x_range * 0.1)
    ax.set_ylim(data["Y"].min() - y_range * 0.1, data["Y"].max() + y_range * 0.1)

    # Scale radius based on map range to avoid overlaps
    max_radius_factor = st.sidebar.slider("Max Pie Chart Size", 0.01, 0.2, 0.08)
    max_radius = min(x_range, y_range) * max_radius_factor

    # --- 3. Draw Pie Charts for each Well ---
    for well_name, well_info in well_data.sort_values(by='TotalThickness', ascending=False).iterrows():
        x, y = well_info['X'], well_info['Y']
        total_thickness = well_info['TotalThickness']

        if total_thickness == 0:
            continue

        layers_in_well = data[data['Well'] == well_name]
        start_angle = 0
        
        # Scale radius of this specific pie based on its total thickness
        # The thickest well gets the max_radius, others are scaled down
        radius = max_radius * (total_thickness / max_filtered_thickness)

        for _, layer_row in layers_in_well.iterrows():
            layer_name = layer_row['Layer']
            thickness = layer_row['Thickness']
            
            proportion = thickness / total_thickness
            angle = proportion * 360
            end_angle = start_angle + angle
            
            color = layer_color_map.get(layer_name, 'gray')
            
            wedge = Wedge(center=(x, y), r=radius, theta1=start_angle, theta2=end_angle, 
                          facecolor=color, edgecolor='black', lw=0.5)
            ax.add_patch(wedge)
            
            start_angle = end_angle

        # Add well name label if toggled on
        if show_well_names:
            ax.text(x, y, well_name, ha="center", va="center", fontsize=8, weight='bold')

    # --- 4. Create and Display Legend ---
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=layer) for layer, color in layer_color_map.items()]
    ax.legend(handles=legend_elements, title="Layers", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    st.pyplot(fig)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Well Data Pie Chart Visualizer")

    st.sidebar.header("Upload and Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded and read successfully!")
            
            # Validate required columns
            required_columns = {'Well', 'X', 'Y', 'Layer', 'Thickness'}
            if not required_columns.issubset(data.columns):
                st.error(f"Error: The uploaded file is missing one or more required columns. Please ensure it contains: {', '.join(required_columns)}")
            else:
                st.subheader("Well Data Preview")
                st.dataframe(data.head())
                plot_well_pie_charts(data)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Awaiting for an Excel file to be uploaded.")

if __name__ == "__main__":
    main()
