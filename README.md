Well Layer Pie Chart Visualizer – App Summary

Purpose
This Streamlit application visualizes well-based properties as layered pie charts positioned on a map using well coordinates. 
It is designed to support spatial interpretation of stratigraphic contributions and well-level totals for any numeric reservoir 
or production property.
Required Input Format
The uploaded Excel file must contain the following base columns:
- Well: Well identifier
- X: Well X coordinate
- Y: Well Y coordinate
- Layer: Stratigraphic layer name
In addition, at least one numeric property column (any name) must exist and can be selected for visualization.
Key Features
1. Property-Agnostic Visualization
Any numeric column can be selected as the plotted property. The app is not hard-coded to thickness or any specific parameter.
2. Two Visualization Modes
Tab 1 – Fixed Pie Size (Per-Well Composition)
All wells are plotted with equal pie size. Slice angles represent the relative contribution of layers within each well.
This mode is best used to compare layer proportions independent of well magnitude.
Tab 2 – Scaled Pie Size (Well-to-Well Comparison)
Pie size scales with total property per well. Larger wells appear larger, smaller wells appear smaller.
An optional filter allows focusing on wells within a specific total property range.
3. Stratigraphic Layer Ordering
Users explicitly define the stratigraphic order of layers. All wells follow the same slice sequence and legend order,
ensuring geological consistency and easy cross-well comparison.
4. Well Label Offset Control
To reduce clutter in dense well maps, well names can be:
- Placed at the pie center
- Offset above the pie
- Offset radially away from the field center
Label offset scales with pie radius to maintain readability.
5. Interactive Styling
- Per-layer color selection
- Adjustable pie sizes
- Toggleable well labels
- Dynamic filtering in scaled mode
Typical Use Cases
- Net pay or thickness distribution by layer
- Layer contribution to well productivity
- Spatial comparison of reservoir quality indicators
- Field-wide screening and QC of stratigraphic data

**Notes**
  
Rows with missing coordinates, layer, or property values are automatically excluded.
Multiple rows per well/layer are aggregated.
The app is intended for exploratory visualization and interpretation.

