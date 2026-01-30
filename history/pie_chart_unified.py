import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_BASE_COLS = ["Well", "X", "Y", "Layer"]


def _validate_base_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def _get_numeric_candidate_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns that are likely numeric properties (exclude base cols).
    We also accept columns that can be coerced to numeric with a reasonable success rate.
    """
    candidates = [c for c in df.columns if c not in REQUIRED_BASE_COLS]

    numeric_cols = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        non_na_ratio = s.notna().mean() if len(s) else 0
        # If at least 70% can be numeric, treat as numeric candidate
        if non_na_ratio >= 0.7:
            numeric_cols.append(c)

    # Also include already-numeric dtype columns (even if ratio < 0.7 due to NaNs)
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in numeric_cols:
            numeric_cols.append(c)

    return numeric_cols


def _prepare_dataframe(df: pd.DataFrame, prop_col: str) -> pd.DataFrame:
    """
    Ensures required columns exist, coerces X/Y/prop to numeric, drops invalid rows.
    """
    out = df.copy()

    # Coerce coordinates & property to numeric
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out[prop_col] = pd.to_numeric(out[prop_col], errors="coerce")

    # Clean strings
    out["Well"] = out["Well"].astype(str).str.strip()
    out["Layer"] = out["Layer"].astype(str).str.strip()

    # Drop rows with missing essentials
    out = out.dropna(subset=["Well", "Layer", "X", "Y", prop_col])

    return out


def _layer_color_controls(unique_layers: list[str], key_prefix: str):
    """
    Create stable color pickers for layers and return a dict layer->color.
    """
    st.sidebar.subheader("Layer Colors")
    default_colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_layers), 1)))

    layer_color_map = {}
    for i, layer in enumerate(unique_layers):
        default_hex = "#%02x%02x%02x" % (
            int(default_colors[i % len(default_colors)][0] * 255),
            int(default_colors[i % len(default_colors)][1] * 255),
            int(default_colors[i % len(default_colors)][2] * 255),
        )
        color = st.sidebar.color_picker(
            f"Color for {layer}",
            value=default_hex,
            key=f"{key_prefix}_color_{layer}",
        )
        layer_color_map[layer] = color
    return layer_color_map


def _compute_well_totals(df: pd.DataFrame, prop_col: str) -> pd.DataFrame:
    """
    Returns per-well totals and coordinates.
    """
    well_data = (
        df.groupby("Well")
        .agg({"X": "first", "Y": "first", prop_col: "sum"})
        .rename(columns={prop_col: "TotalProperty"})
    )
    return well_data


def _plot_pies(
    df: pd.DataFrame,
    prop_col: str,
    layer_color_map: dict,
    show_well_names: bool,
    radius_mode: str,
    radius_factor: float,
    max_radius_factor: float,
    filter_range: tuple[float, float] | None,
):
    """
    radius_mode:
      - "fixed": one constant radius for all wells (radius_factor)
      - "scaled": radius scales per-well based on TotalProperty (max_radius_factor)
    """
    if df.empty:
        st.warning("No valid rows to plot after cleaning. Check your data.")
        return

    well_data = _compute_well_totals(df, prop_col)

    # Optional filter (only meaningful for scaled tab, but can be used anywhere)
    if filter_range is not None and not well_data.empty:
        lo, hi = filter_range
        well_data = well_data[(well_data["TotalProperty"] >= lo) & (well_data["TotalProperty"] <= hi)]

    if well_data.empty:
        st.warning("No wells left after filtering.")
        return

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Well Layer Pie Chart Map ({prop_col})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal", adjustable="box")

    x_range = df["X"].max() - df["X"].min()
    y_range = df["Y"].max() - df["Y"].min()
    # Avoid zero ranges
    x_range = x_range if x_range != 0 else 1.0
    y_range = y_range if y_range != 0 else 1.0

    ax.set_xlim(df["X"].min() - x_range * 0.1, df["X"].max() + x_range * 0.1)
    ax.set_ylim(df["Y"].min() - y_range * 0.1, df["Y"].max() + y_range * 0.1)

    # Radius config
    if radius_mode == "fixed":
        base_radius = min(x_range, y_range) * radius_factor
        max_filtered_total = None
    else:
        max_radius = min(x_range, y_range) * max_radius_factor
        max_filtered_total = well_data["TotalProperty"].max() if not well_data.empty else 1.0
        if max_filtered_total == 0:
            max_filtered_total = 1.0

    # Draw pies (largest first for better visibility)
    for well_name, w in well_data.sort_values(by="TotalProperty", ascending=False).iterrows():
        x, y = float(w["X"]), float(w["Y"])
        total_val = float(w["TotalProperty"])
        if total_val == 0:
            continue

        # Determine radius for this well
        if radius_mode == "fixed":
            radius = base_radius
        else:
            radius = max_radius * (total_val / max_filtered_total)

        layers_in_well = df[df["Well"] == well_name]
        start_angle = 0.0

        # Build wedges
        for _, row in layers_in_well.iterrows():
            layer_name = row["Layer"]
            v = float(row[prop_col])

            if v <= 0:
                continue

            proportion = v / total_val
            angle = proportion * 360.0
            end_angle = start_angle + angle

            color = layer_color_map.get(layer_name, "gray")
            wedge = Wedge(
                center=(x, y),
                r=radius,
                theta1=start_angle,
                theta2=end_angle,
                facecolor=color,
                edgecolor="black",
                lw=0.5,
            )
            ax.add_patch(wedge)

            start_angle = end_angle

        if show_well_names:
            ax.text(x, y, str(well_name), ha="center", va="center", fontsize=8, weight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=layer)
        for layer, color in layer_color_map.items()
    ]
    ax.legend(handles=legend_elements, title="Layers", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Well Data Pie Chart Visualizer (Unified)")

    st.sidebar.header("Upload and Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file is None:
        st.info("Awaiting an Excel file to be uploaded.")
        return

    # Read
    try:
        data_raw = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded and read successfully!")
    except Exception as e:
        st.error(f"An error occurred while reading the Excel file: {e}")
        return

    # Validate base columns
    ok, missing = _validate_base_columns(data_raw)
    if not ok:
        st.error(
            "Missing required columns: "
            + ", ".join(missing)
            + "\n\nRequired base columns are: "
            + ", ".join(REQUIRED_BASE_COLS)
            + "\nAdd at least one numeric property column to plot (any name you want)."
        )
        st.subheader("Uploaded Data Preview")
        st.dataframe(data_raw.head(50))
        return

    # Choose property
    numeric_candidates = _get_numeric_candidate_columns(data_raw)
    if not numeric_candidates:
        st.error(
            "I couldn't detect any numeric property column.\n\n"
            "Keep these base columns: Well, X, Y, Layer\n"
            "Then add at least one numeric property column (e.g., Thickness, Porosity, Kh, NetPay, etc.)."
        )
        st.subheader("Uploaded Data Preview")
        st.dataframe(data_raw.head(50))
        return

    prop_col = st.sidebar.selectbox("Select Property Column to Plot", numeric_candidates)

    # Prepare/clean
    data = _prepare_dataframe(data_raw, prop_col)
    if data.empty:
        st.error("After cleaning (numeric coercion + dropping invalid rows), no data remained.")
        st.subheader("Uploaded Data Preview (Raw)")
        st.dataframe(data_raw.head(50))
        return

    st.subheader("Data Preview (Cleaned)")
    st.dataframe(data.head(50))

    # Shared controls
    st.sidebar.subheader("Display Options")
    show_well_names = st.sidebar.checkbox("Show Well Names", value=True)

    unique_layers = sorted(data["Layer"].unique().tolist())
    layer_color_map = _layer_color_controls(unique_layers, key_prefix="global")

    # Tabs
    tab1, tab2 = st.tabs(["Per-Well Fixed Pie Size", "Offset/Scaled Pie Size (by Well Total)"])

    with tab1:
        st.markdown(
            f"### Fixed-size pie per well\n"
            f"Pies are the **same radius**; slice sizes are based on **{prop_col}** per layer in each well."
        )
        radius_factor = st.sidebar.slider("Pie Chart Size (Fixed)", 0.01, 0.2, 0.05, key="fixed_radius_factor")

        _plot_pies(
            df=data,
            prop_col=prop_col,
            layer_color_map=layer_color_map,
            show_well_names=show_well_names,
            radius_mode="fixed",
            radius_factor=radius_factor,
            max_radius_factor=0.08,
            filter_range=None,
        )

    with tab2:
        st.markdown(
            f"### Scaled-size pie per well + filter\n"
            f"Pie radius is scaled by **total {prop_col} per well** (largest well gets max size)."
        )

        # Filter by per-well totals
        well_totals = _compute_well_totals(data, prop_col)
        min_total = float(well_totals["TotalProperty"].min()) if not well_totals.empty else 0.0
        max_total = float(well_totals["TotalProperty"].max()) if not well_totals.empty else 1.0

        st.sidebar.subheader("Scaled Tab Controls")
        min_filter, max_filter = st.sidebar.slider(
            f"Filter by Total Well {prop_col}",
            min_value=min_total,
            max_value=max_total,
            value=(min_total, max_total),
            key="scaled_filter",
        )

        max_radius_factor = st.sidebar.slider("Max Pie Chart Size (Scaled)", 0.01, 0.2, 0.08, key="scaled_max_radius")

        _plot_pies(
            df=data,
            prop_col=prop_col,
            layer_color_map=layer_color_map,
            show_well_names=show_well_names,
            radius_mode="scaled",
            radius_factor=0.05,
            max_radius_factor=max_radius_factor,
            filter_range=(min_filter, max_filter),
        )


if __name__ == "__main__":
    main()
