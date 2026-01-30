import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import numpy as np

# -----------------------------
# Constants
# -----------------------------
REQUIRED_BASE_COLS = ["Well", "X", "Y", "Layer"]


# -----------------------------
# Helpers
# -----------------------------
def _validate_base_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def _get_numeric_candidate_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns that are likely numeric properties (exclude base cols).
    We accept columns that can be coerced to numeric with a reasonable success rate.
    """
    candidates = [c for c in df.columns if c not in REQUIRED_BASE_COLS]

    numeric_cols = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        non_na_ratio = s.notna().mean() if len(s) else 0
        if non_na_ratio >= 0.7:
            numeric_cols.append(c)

    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in numeric_cols:
            numeric_cols.append(c)

    return numeric_cols


def _prepare_dataframe(df: pd.DataFrame, prop_col: str) -> pd.DataFrame:
    """
    Ensures required columns exist, coerces X/Y/prop to numeric, drops invalid rows.
    """
    out = df.copy()

    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out[prop_col] = pd.to_numeric(out[prop_col], errors="coerce")

    out["Well"] = out["Well"].astype(str).str.strip()
    out["Layer"] = out["Layer"].astype(str).str.strip()

    out = out.dropna(subset=["Well", "Layer", "X", "Y", prop_col])
    return out


def _layer_color_controls(unique_layers: list[str], key_prefix: str):
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
    well_data = (
        df.groupby("Well")
        .agg({"X": "first", "Y": "first", prop_col: "sum"})
        .rename(columns={prop_col: "TotalProperty"})
    )
    return well_data


def _build_layer_order(unique_layers: list[str]) -> list[str]:
    """
    Streamlit-friendly ordering: user selects ordered subset; remainder appended.
    """
    st.sidebar.subheader("Layer Ordering (Stratigraphic)")
    st.sidebar.caption(
        "Pick layers in the exact stratigraphic order you want slices to follow (top→bottom or bottom→top). "
        "Unselected layers will be appended at the end."
    )
    chosen = st.sidebar.multiselect(
        "Ordered layers (in the order you click them)",
        options=unique_layers,
        default=unique_layers,  # start with all selected (sorted), user can adjust
        key="layer_order_multiselect",
    )
    # Preserve the order as chosen in the multiselect (Streamlit returns in selection order)
    remaining = [l for l in unique_layers if l not in chosen]
    return chosen + remaining


def _label_position(
    x: float,
    y: float,
    radius: float,
    mode: str,
    factor: float,
    field_center: tuple[float, float],
):
    """
    Returns (lx, ly, ha, va) for label placement.
    mode:
      - "center": at (x,y)
      - "above": (x, y + radius*factor)
      - "radial": push away from field center by radius*factor
    """
    if mode == "center":
        return x, y, "center", "center"

    if mode == "above":
        return x, y + radius * factor, "center", "bottom"

    # radial
    cx, cy = field_center
    dx = x - cx
    dy = y - cy
    n = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / n, dy / n
    return x + ux * radius * factor, y + uy * radius * factor, "center", "center"


def _plot_pies(
    df: pd.DataFrame,
    prop_col: str,
    layer_color_map: dict,
    layer_order: list[str],
    show_well_names: bool,
    label_mode: str,          # "center" | "above" | "radial"
    label_offset_factor: float,
    radius_mode: str,         # "fixed" | "scaled"
    radius_factor: float,
    max_radius_factor: float,
    filter_range: tuple[float, float] | None,
):
    if df.empty:
        st.warning("No valid rows to plot after cleaning. Check your data.")
        return

    well_data = _compute_well_totals(df, prop_col)

    # Optional filter
    if filter_range is not None and not well_data.empty:
        lo, hi = filter_range
        well_data = well_data[(well_data["TotalProperty"] >= lo) & (well_data["TotalProperty"] <= hi)]

    if well_data.empty:
        st.warning("No wells left after filtering.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Well Layer Pie Chart Map ({prop_col})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal", adjustable="box")

    x_min, x_max = float(df["X"].min()), float(df["X"].max())
    y_min, y_max = float(df["Y"].min()), float(df["Y"].max())
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0

    ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    # Field center for radial label offset
    field_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    # Radius config
    if radius_mode == "fixed":
        base_radius = min(x_range, y_range) * radius_factor
        max_filtered_total = None
    else:
        max_radius = min(x_range, y_range) * max_radius_factor
        max_filtered_total = float(well_data["TotalProperty"].max()) if not well_data.empty else 1.0
        if max_filtered_total == 0:
            max_filtered_total = 1.0

    # Draw pies (largest first)
    for well_name, w in well_data.sort_values(by="TotalProperty", ascending=False).iterrows():
        x, y = float(w["X"]), float(w["Y"])
        total_val = float(w["TotalProperty"])
        if total_val <= 0:
            continue

        radius = base_radius if radius_mode == "fixed" else (max_radius * (total_val / max_filtered_total))

        # Subset per well
        wdf = df[df["Well"] == well_name].copy()

        # Aggregate per layer (in case multiple rows per layer)
        layer_vals = (
            wdf.groupby("Layer")[prop_col]
            .sum()
            .reindex(layer_order)
            .dropna()
        )

        # If nothing left after ordering
        if layer_vals.empty:
            continue

        start_angle = 0.0
        for layer_name, v in layer_vals.items():
            v = float(v)
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
            lx, ly, ha, va = _label_position(
                x=x,
                y=y,
                radius=radius,
                mode=label_mode,
                factor=label_offset_factor,
                field_center=field_center,
            )
            ax.text(lx, ly, str(well_name), ha=ha, va=va, fontsize=8, weight="bold")

    # Legend in the chosen stratigraphic order (only layers that exist)
    legend_layers = [l for l in layer_order if l in layer_color_map]
    legend_elements = [Patch(facecolor=layer_color_map[l], edgecolor="black", label=l) for l in legend_layers]
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
        st.subheader("Uploaded Data Preview (Raw)")
        st.dataframe(data_raw.head(50))
        return

    # Choose property column
    numeric_candidates = _get_numeric_candidate_columns(data_raw)
    if not numeric_candidates:
        st.error(
            "I couldn't detect any numeric property column.\n\n"
            "Keep these base columns: Well, X, Y, Layer\n"
            "Then add at least one numeric property column (e.g., Thickness, Porosity, Kh, NetPay, etc.)."
        )
        st.subheader("Uploaded Data Preview (Raw)")
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

    # Display + label offset controls
    st.sidebar.subheader("Display Options")
    show_well_names = st.sidebar.checkbox("Show Well Names", value=True)

    label_mode_ui = st.sidebar.selectbox(
        "Well Label Placement",
        options=["Center (on pie)", "Offset above pie", "Offset radially (away from field center)"],
        index=1,
    )
    label_mode = {
        "Center (on pie)": "center",
        "Offset above pie": "above",
        "Offset radially (away from field center)": "radial",
    }[label_mode_ui]

    label_offset_factor = st.sidebar.slider(
        "Label Offset Factor (x radius)",
        min_value=0.8,
        max_value=3.0,
        value=1.4,
        step=0.1,
        help="Higher = label further from the pie edge (radius-aware).",
    )

    # Layer ordering + colors
    unique_layers = sorted(data["Layer"].unique().tolist())
    layer_order = _build_layer_order(unique_layers)
    layer_color_map = _layer_color_controls(unique_layers, key_prefix="global")

    # Tabs
    tab1, tab2 = st.tabs(["Per-Well Fixed Pie Size", "Offset/Scaled Pie Size (by Well Total)"])

    with tab1:
        st.markdown(
            f"### Fixed-size pie per well\n"
            f"Pies are the **same radius**; slice sizes follow **{prop_col}** per layer.\n\n"
            f"**Slice order is stratigraphic** using your selected Layer Ordering."
        )
        radius_factor = st.sidebar.slider(
            "Pie Size (Fixed Tab)",
            0.01, 0.2, 0.05, 0.01,
            key="fixed_radius_factor"
        )

        _plot_pies(
            df=data,
            prop_col=prop_col,
            layer_color_map=layer_color_map,
            layer_order=layer_order,
            show_well_names=show_well_names,
            label_mode=label_mode,
            label_offset_factor=label_offset_factor,
            radius_mode="fixed",
            radius_factor=radius_factor,
            max_radius_factor=0.08,
            filter_range=None,
        )

    with tab2:
        st.markdown(
            f"### Scaled-size pie per well + filter\n"
            f"Pie radius is scaled by **total {prop_col} per well** (largest well gets max size).\n\n"
            f"**Slice order is stratigraphic** using your selected Layer Ordering."
        )

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

        max_radius_factor = st.sidebar.slider(
            "Max Pie Size (Scaled Tab)",
            0.01, 0.2, 0.08, 0.01,
            key="scaled_max_radius"
        )

        _plot_pies(
            df=data,
            prop_col=prop_col,
            layer_color_map=layer_color_map,
            layer_order=layer_order,
            show_well_names=show_well_names,
            label_mode=label_mode,
            label_offset_factor=label_offset_factor,
            radius_mode="scaled",
            radius_factor=0.05,
            max_radius_factor=max_radius_factor,
            filter_range=(min_filter, max_filter),
        )


if __name__ == "__main__":
    main()
