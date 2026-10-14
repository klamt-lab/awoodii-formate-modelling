"""Build MDF scan figures with concentration, ratio, and bottleneck panels."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from typing import List, Optional, Dict, Any
import logging
from mymodelingpackage.analysis.utils import generate_shades, parse_dict_from_str, parse_list_from_str, calculate_global_y_range


def _clean_empty_subplots(fig: go.Figure, rows: int, cols: int):
    """
    Hides titles and axes for any subplot in the grid that contains no data traces.
    This ensures that excluded datasets appear as blank white space, not empty boxes with titles.
    """
    # 1. Identify which subplots actually have data
    # We store (row, col) tuples of populated subplots
    has_data = set()
    
    for trace in fig.data:
        # Get axis name used by this trace (e.g., 'x', 'x2', 'x10')
        xaxis = trace.xaxis or 'x'
        
        # Convert axis name to linear index (1-based)
        # 'x' -> 1, 'x2' -> 2
        idx_str = xaxis.replace('x', '')
        idx = int(idx_str) if idx_str else 1
        
        # Convert linear index to (Row, Col) grid coordinates
        # Formula assumes row-major order (standard for make_subplots)
        r = (idx - 1) // cols + 1
        c = (idx - 1) % cols + 1
        
        has_data.add((r, c))

    # 2. Iterate through grid and hide elements for empty spots
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if (r, c) not in has_data:
                # A. Hide Axis Lines, Ticks, and Labels
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                
                # B. Hide Subplot Title
                # Titles are stored in layout.annotations in the same order as subplots
                annot_idx = (r - 1) * cols + (c - 1)
                
                # Safety check: ensure we don't crash if annotations are missing
                if fig.layout.annotations and annot_idx < len(fig.layout.annotations):
                    fig.layout.annotations[annot_idx].text = ""


def _setup_plot_layout(layout_mode: str, datasets_config: list, secondary_config: list, primary_plot_options: dict = None) -> dict:
    """
    Prepares layout parameters for the comparison plot based on a list-based secondary analysis configuration.
    
    Returns a dictionary with all necessary layout parameters.
    """
    logging.debug(f"Setting up plot layout in mode: '{layout_mode}'")
    # --- 1. Determine Row and Column counts ---
    # The number of rows is now dynamic: 1 for the primary plot + 1 for each secondary plot requested.
    num_rows = 1 + len(secondary_config)

    primary_plot_options = primary_plot_options or {}

    color_palette = px.colors.qualitative.G10
    trace_name_map = {}
    for dataset_config in datasets_config:
        dataset_name = str(dataset_config["name"])
        # Legend labels can be configured directly. Otherwise grouped overlays
        # use the model/group title plus the analysis label so traces remain
        # readable after several analyses are plotted into the same column.
        if dataset_config.get("legend_label"):
            trace_name = str(dataset_config["legend_label"])
        elif dataset_config.get("trace_label"):
            trace_label = str(dataset_config["trace_label"])
            if layout_mode == "grouped_overlay":
                group_label = (
                    dataset_config.get("column_title")
                    or dataset_config.get("title")
                    or dataset_config.get("column_group")
                    or dataset_name
                )
                trace_name = f"{group_label}: {trace_label}"
            else:
                trace_name = trace_label
        else:
            trace_name = dataset_name
        trace_name_map[dataset_config["name"]] = trace_name
    color_map = {}

    # Column setup remains the same
    if layout_mode == "overlay":
        num_cols = 1
        primary_titles = [primary_plot_options.get("title", f"Overlay Plot ({len(datasets_config)} datasets)")]
        col_map = {d["name"]: 1 for d in datasets_config}
        color_map = {
            d["name"]: d.get("color", color_palette[i % len(color_palette)])
            for i, d in enumerate(datasets_config)
        }
    elif layout_mode == "side_by_side":
        num_cols = len(datasets_config)
        primary_titles = []
        for dataset_config in datasets_config:
            title = dataset_config.get("title", dataset_config["name"])
            # Side-by-side columns describe one dataset each, so the dataset
            # subtitle can live directly under the subplot title.
            if dataset_config.get("subtitle"):
                title = f"{title}<br>{dataset_config['subtitle']}"
            primary_titles.append(str(title))
        col_map = {d["name"]: i + 1 for i, d in enumerate(datasets_config)}
        color_map = {
            d["name"]: d.get("color", color_palette[i % len(color_palette)])
            for i, d in enumerate(datasets_config)
        }
    elif layout_mode == "grouped_overlay":
        group_order = []
        group_members = {}
        group_titles = {}

        for dataset_config in datasets_config:
            group_key = str(dataset_config.get("column_group") or dataset_config.get("group") or dataset_config["name"])
            if group_key not in group_members:
                group_order.append(group_key)
                group_members[group_key] = []
                title = dataset_config.get("column_title") or dataset_config.get("title") or group_key
                subtitle = dataset_config.get("column_subtitle") or dataset_config.get("subtitle")
                # Grouped overlays describe one model per column. The analyses
                # become traces, while the column title stays at model level.
                if subtitle:
                    title = f"{title}<br>{subtitle}"
                group_titles[group_key] = str(title)
            group_members[group_key].append(dataset_config)

        num_cols = len(group_order)
        primary_titles = [group_titles[group_key] for group_key in group_order]
        group_col_map = {group_key: index + 1 for index, group_key in enumerate(group_order)}
        col_map = {}

        for group_index, group_key in enumerate(group_order):
            members = group_members[group_key]
            for dataset_config in members:
                col_map[dataset_config["name"]] = group_col_map[group_key]

            base_color = next(
                (
                    member.get("group_color") or member.get("color")
                    for member in members
                    if member.get("group_color") or member.get("color")
                ),
                color_palette[group_index % len(color_palette)],
            )
            auto_color_members = [member for member in members if not member.get("trace_color")]
            auto_colors = iter(generate_shades(base_color, len(auto_color_members), max_lightness_factor=0.85))
            for member in members:
                color_map[member["name"]] = member.get("trace_color") or next(auto_colors)
    else:
        raise ValueError("layout_mode must be 'overlay', 'side_by_side', or 'grouped_overlay'")
    logging.debug(f"  - Primary plot grid: 1 row x {num_cols} cols")
    # --- 2. Build Subplot Titles ---
    # Start with the title for the primary plot (row 1)
    # This list will be flattened by make_subplots, so we build it row by row.
    subplot_titles = primary_titles * num_rows # Temporarily fill all rows

    # Overwrite the titles for the secondary analysis rows
    for i, panel_config in enumerate(secondary_config):
        row_idx = i + 1 # This is the row index in the secondary plots (0-indexed)
        # The title for this row is repeated for each column
        row_titles = [panel_config.get("title", f"Secondary Analysis {i+1}")] * num_cols
        # Replace the placeholder titles for this row
        start_idx = row_idx * num_cols
        end_idx = start_idx + num_cols
        subplot_titles[start_idx:end_idx] = row_titles

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "col_map": col_map,
        "color_map": color_map,
        "trace_name_map": trace_name_map,
        "subplot_titles": subplot_titles,
    }


def add_min_max_traces(fig, df_x, data_series, prefix, unique_legend_name, row, col, color):
    """
    Internal helper to add min/max traces.
    *** CORRECTED: The hover name is simpler and it uses the unique legend group passed in. ***
    """
    min_vals = pd.to_numeric(data_series.apply(lambda d: d.get('min') if isinstance(d, dict) else None), errors='coerce')
    max_vals = pd.to_numeric(data_series.apply(lambda d: d.get('max') if isinstance(d, dict) else None), errors='coerce')

    if min_vals.dropna().empty and max_vals.dropna().empty:
        return

    # Link this secondary trace to its unique primary trace for toggling
    legend_group = unique_legend_name
    
    hover_name = f"{prefix} ({unique_legend_name})"
    
    fig.add_trace(go.Scatter(
        x=df_x, y=max_vals, mode='lines', 
        name=f"{hover_name} max",
        line=dict(color=color, width=2),
        legendgroup=legend_group,
        showlegend=False 
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=df_x, y=min_vals, mode='lines', 
        name=f"{hover_name} min",
        line=dict(color=color, width=2, dash='dash'),
        legendgroup=legend_group,
        showlegend=False 
    ), row=row, col=col)


def plot_secondary_analysis_trends_grouped(
    fig: go.Figure,
    df: pd.DataFrame, 
    scan_variable: str,
    dataset_column: str,
    col_map: Dict[str, int],
    color_map: Dict[str, str],
    secondary_config: list,
    sorting_key_column: Optional[str] = None,
    trace_name_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Populates the secondary analysis panels (rows 2+) of the master figure with trend lines.

    This function iterates through every dataset present in the DataFrame. For each dataset,
    it iterates through the configured secondary analysis panels (FVA, CVA, Ratios, Bottlenecks).
    It handles data parsing (converting stringified dicts/lists to numbers), visual styling
    (colors, shading for variations), and trace grouping.

    Key Features:
    1. **Dynamic Shading**: If `sorting_key_column` is provided, it splits a dataset into 
       multiple shade lines (e.g., varying H2 concentrations) using a color gradient.
    2. **Data Parsing**: Safely unpacks columns like 'FVA_Results' which are stored as 
       string representations of Python dictionaries in the CSV.
    3. **Conditional Exclusion**: Allows specific datasets to be excluded from specific 
       subplots via the 'exclude_datasets' key in `secondary_config`.

    Args:
        fig (go.Figure): The Plotly figure object to append traces to.
        df (pd.DataFrame): The master DataFrame containing all simulation results.
        scan_variable (str): The name of the independent variable (x-axis), e.g., 'biomass'.
                             Used to construct the x-axis column name (`{scan_variable}_target`).
        dataset_column (str): The column name identifying the dataset (e.g., 'dataset_name').
        col_map (Dict[str, int]): Mapping of dataset names to specific subplot column indices.
        color_map (Dict[str, str]): Mapping of dataset names to their base hex colors.
        secondary_config (List[Dict[str, Any]]): Configuration list for the secondary rows. 
            Each dict defines a panel type ('FVA', 'CVA', etc.) and items to plot.
            Can include an 'exclude_datasets' list to hide specific data.
        sorting_key_column (Optional[str]): Column used to differentiate variations within 
            a single dataset (e.g., 'var_h2_c_max'). If present, generates shaded lines.

    Returns:
        go.Figure: The modified Plotly figure with added traces.
    """
    # Construct the x-axis column name based on the scan variable (e.g., "biomass_target")
    x_col = f"{scan_variable}_target"
    trace_name_map = trace_name_map or {}
    logging.debug(f"Starting secondary analysis plotting. X-axis column: '{x_col}'")
    
    # --- Outer Loop: Iterate through each unique dataset ---
    # We group by dataset first to maintain consistent coloring and column placement.
    for dataset_name, group_df in df.groupby(dataset_column):
        logging.debug(f"Processing dataset: '{dataset_name}'")
        # 1. Prepare Data for this Dataset
        # Sort by x-axis to ensure line plots are drawn sequentially without zig-zags.
        group_df = group_df.sort_values(by=x_col).copy()
        # Retrieve layout and style info for this dataset
        target_col = col_map.get(dataset_name, 1)
        base_color = color_map.get(dataset_name)
        base_legend_name = trace_name_map.get(dataset_name, dataset_name)

        # 2. Determine Shading Logic (Variations within a dataset)
        # Check if the sorting column exists, has data, and has more than 1 unique value.
        has_shades = sorting_key_column and sorting_key_column in group_df.columns and group_df[sorting_key_column].nunique() > 1
        
        shade_iterator = []
        if has_shades:
            # Generate a list of (value, color) tuples for each variation
            unique_shade_values = sorted(group_df[sorting_key_column].dropna().unique())
            num_shades = len(unique_shade_values)
            logging.debug(f"  -> Shading detected. Generating {num_shades} shades for '{sorting_key_column}'.")

            # Generate color gradient (dark to light base color)
            color_palette = generate_shades(base_color, num_shades, max_lightness_factor=0.85)
            shade_iterator = zip(unique_shade_values, color_palette)
        else:
            # No shading: Use a single iterator with None value and the base color
            logging.debug(f"  -> No shading. Using base color: {base_color}")
            shade_iterator = [(None, base_color)]

        # --- Middle Loop: Iterate through shades (or just once if no shades) ---
        for shade_value, trace_color in shade_iterator:
            unique_legend_name = ""

            # 3. Filter Subset and Generate Legend Name
            if has_shades:
                shade_subset = group_df[group_df[sorting_key_column] == shade_value]
                # Create a clean string for the legend (e.g., "TCOSA (h2=1.0e-5)")
                # Replicated from the primary plot function
                try:
                    # Strip prefixes like 'var_' or suffixes like '_max' for cleaner display
                    short_key = sorting_key_column.replace('var_', '').replace('_min','').replace('_max','')
                    unique_legend_name = f"{base_legend_name} ({short_key}={shade_value:.1e})"
                except (TypeError, ValueError):
                    unique_legend_name = f"{base_legend_name} ({short_key}={shade_value})"
            else:
                shade_subset = group_df
                unique_legend_name = base_legend_name # No variations, group is the dataset itself

            if shade_subset.empty:
                logging.debug(f"  -> Skipping empty subset for shade: {shade_value}")
                continue

            # --- Inner Loop: Iterate through configured Secondary Panels ---
            # This corresponds to rows 2, 3, 4, etc. in the subplot grid.
            for i, panel_config in enumerate(secondary_config):
                # Row index calculation: 
                # Row 1 is the primary plot. 
                # Secondary config index 0 goes to Row 2, Index 1 to Row 3, etc.
                target_row = i + 2
                panel_type = panel_config.get("type")
                panel_title = panel_config.get("title", "Untitled")

                # --- 4. Exclusion Logic ---
                # Check if this specific dataset is blacklisted for this specific panel.
                # We use substring matching to handle HTML tags in dataset names.
                exclusions = panel_config.get("exclude_datasets", [])
                if exclusions and any(excl in dataset_name for excl in exclusions):
                    # Check if any exclusion string is a substring of the current dataset name
                    is_excluded = any(excl in dataset_name for excl in exclusions)
                    if is_excluded:
                        logging.debug(f"    -> [Excluded] Dataset '{dataset_name}' skipped for panel '{panel_title}' (matched exclusion list).")
                        continue

                logging.debug(f"    -> Adding traces for panel '{panel_type}' ({panel_title}) at Row {target_row}, Col {target_col}")
                
                # --- 5. Data Processing per Panel Type ---
                
                # Case A: Flux Variability Analysis (FVA)
                # Data structure: Stringified dictionary {'R1': {'min': x, 'max': y}, ...}
                if panel_type == "FVA" and 'FVA_Results' in shade_subset.columns:
                    reactions_to_plot = panel_config.get("reactions_to_show", [])
                    # Parse the stringified JSON into actual dictionaries
                    parsed = shade_subset['FVA_Results'].apply(parse_dict_from_str)
                    # Expand into columns for easy access
                    expanded = pd.DataFrame(parsed.tolist(), index=parsed.index)
                    for rxn_id in reactions_to_plot:
                        if rxn_id in expanded.columns:
                            add_min_max_traces(
                                fig, shade_subset[x_col], expanded[rxn_id], 
                                prefix=rxn_id, 
                                unique_legend_name=unique_legend_name, 
                                row=target_row, col=target_col, color=trace_color
                            )
                        else:
                            logging.debug(f"      -> Warning: Reaction '{rxn_id}' not found in FVA results.")
                
                # Case B: Concentration Variability Analysis (CVA)
                elif panel_type == "CVA" and 'CVA_Results' in shade_subset.columns:
                    mets_to_plot = panel_config.get("metabolites_to_show", [])
                    parsed = shade_subset['CVA_Results'].apply(parse_dict_from_str)
                    expanded = pd.DataFrame(parsed.tolist(), index=parsed.index)
                    for met_id in mets_to_plot:
                         if met_id in expanded.columns:
                            add_min_max_traces(
                                fig, shade_subset[x_col], expanded[met_id], 
                                prefix=met_id, 
                                unique_legend_name=unique_legend_name, 
                                row=target_row, col=target_col, color=trace_color
                            )
                         else:
                            logging.debug(f"      -> Warning: Metabolite '{met_id}' not found in CVA results.")

                # Case C: Metabolite Ratios
                elif panel_type == "Metabolite_Ratios" and 'Metabolite_Ratios' in shade_subset.columns:
                    ratios_to_plot = panel_config.get("ratios_to_show", [])
                    parsed = shade_subset['Metabolite_Ratios'].apply(parse_dict_from_str)
                    expanded = pd.DataFrame(parsed.tolist(), index=parsed.index)
                    for ratio_id in ratios_to_plot:
                         if ratio_id in expanded.columns:
                            add_min_max_traces(
                                fig, shade_subset[x_col], expanded[ratio_id], 
                                prefix=ratio_id, 
                                unique_legend_name=unique_legend_name, 
                                row=target_row, col=target_col, color=trace_color
                            )
                         else:
                             logging.debug(f"      -> Warning: Ratio '{ratio_id}' not found in ratio results.")

                # Case D: Bottlenecks
                # Existing perturbation analysis is a stringified list. The new
                # MILP set analysis is a stringified dict with a minimal count.
                elif panel_type in {"Bottlenecks", "Minimal_Bottleneck_Sets"}:
                    if panel_type == "Bottlenecks" and 'Bottleneck_Reactions' in shade_subset.columns:
                        counts = shade_subset['Bottleneck_Reactions'].apply(lambda x: len(parse_list_from_str(x)))
                        fig.add_trace(go.Scatter(
                            x=shade_subset[x_col],
                            y=counts,
                            mode='lines+markers',
                            name=f"Bottlenecks ({unique_legend_name})",
                            line=dict(shape='hv', color=trace_color),
                            customdata=shade_subset['Bottleneck_Reactions'],
                            hovertemplate=(
                                "%{x}<br>"
                                "<b>Perturbation bottleneck count:</b> %{y}<br>"
                                "<b>Reactions:</b> %{customdata}<extra></extra>"
                            ),
                            legendgroup=unique_legend_name,
                            showlegend=False
                        ), row=target_row, col=target_col)

                    if 'Minimal_Bottleneck_Sets' in shade_subset.columns:
                        parsed_sets = shade_subset['Minimal_Bottleneck_Sets'].apply(parse_dict_from_str)
                        minimal_counts = pd.to_numeric(
                            parsed_sets.apply(
                                lambda d: d.get('minimal_relaxation_count') if isinstance(d, dict) else None
                            ),
                            errors='coerce'
                        )
                        if not minimal_counts.dropna().empty:
                            fig.add_trace(go.Scatter(
                                x=shade_subset[x_col],
                                y=minimal_counts,
                                mode='lines+markers',
                                name=f"Minimal bottleneck sets ({unique_legend_name})",
                                line=dict(shape='hv', color=trace_color, dash='dot'),
                                customdata=shade_subset['Minimal_Bottleneck_Sets'],
                                hovertemplate=(
                                    "%{x}<br>"
                                    "<b>Minimal relaxed reactions:</b> %{y}<br>"
                                    "<b>Set analysis:</b> %{customdata}<extra></extra>"
                                ),
                                legendgroup=unique_legend_name,
                                showlegend=False
                            ), row=target_row, col=target_col)
    return fig


def generate_mdfscan_custom_plot(
    df: pd.DataFrame,
    scan_variable: str,
    datasets_config: list,
    secondary_config: list,
    layout_mode: str,
    plot_title: str,
    primary_plot_options: dict = None,
    global_axis_labels: dict = None,
    show_legend: bool = True,
    sorting_key_column: Optional[str] = None,
    vertical_spacing: float = 0.08,
    font_size: int = 16,
    title_font_size: Optional[int] = None,       
    subplot_title_font_size: Optional[int] = None 
) -> go.Figure:
    """
    Generates a comprehensive, multi-panel comparison plot for MDF scan results.
    This function supports detailed customization of titles and axis labels
    through the plot_config dictionary.
    """
    primary_plot_options = primary_plot_options or {}
    global_axis_labels = global_axis_labels or {}
    primary_y_column = primary_plot_options.get("y_column", "Objective_Value")
    if primary_y_column not in df.columns:
        fallback_y_column = "MinForce"
        logging.warning("Primary VariableScan y-column '%s' is missing; falling back to '%s'.", primary_y_column, fallback_y_column)
        primary_y_column = fallback_y_column

    # --- 1. SETUP: Determine layout and create the subplots figure ---
    layout = _setup_plot_layout(
        layout_mode, datasets_config, secondary_config,
        primary_plot_options=primary_plot_options
    )
    fig = make_subplots(
        rows=layout["num_rows"], cols=layout["num_cols"],
        subplot_titles=layout["subplot_titles"], shared_xaxes=True,
        vertical_spacing=vertical_spacing
    )

    # --- 2. ADD TRACES: Populate the figure with data ---

    # Add Primary VariableScan Plot Traces (Row 1)
    for dataset_cfg in datasets_config:
        dataset_name = dataset_cfg["name"]
        df_subset = df[df['dataset_name'] == dataset_name]
        if df_subset.empty: continue

        target_col = layout["col_map"][dataset_name]
        base_color = layout["color_map"][dataset_name]
        base_legend_name = layout["trace_name_map"].get(dataset_name, dataset_name)

        has_shades = sorting_key_column and sorting_key_column in df_subset.columns and df_subset[sorting_key_column].nunique() > 1
        
        shade_iterator = []
        if has_shades:
            unique_shade_values = sorted(df_subset[sorting_key_column].dropna().unique())
            num_shades = len(unique_shade_values)
            color_palette = generate_shades(base_color, num_shades, max_lightness_factor=0.85)
            shade_iterator = zip(unique_shade_values, color_palette)
        else:
            shade_iterator = [(None, base_color)]

        for shade_value, trace_color in shade_iterator:
            unique_legend_name = ""
            if has_shades:
                variation_subset = df_subset[df_subset[sorting_key_column] == shade_value]
                try:
                    short_key = sorting_key_column.replace('var_', '').replace('_min','').replace('_max','')
                    unique_legend_name = f"{base_legend_name} ({short_key}={shade_value:.1e})"
                except (TypeError, ValueError):
                    unique_legend_name = f"{base_legend_name} ({short_key}={shade_value})"
            else:
                variation_subset = df_subset
                unique_legend_name = base_legend_name
            
            fig.add_trace(go.Scatter(
                x=variation_subset[f"{scan_variable}_target"],
                y=variation_subset[primary_y_column],
                mode='lines+markers',
                name=unique_legend_name,
                legendgroup=unique_legend_name,
                showlegend=True,
                marker=dict(color=trace_color),
                line=dict(color=trace_color)
            ), row=1, col=target_col)

    # Add All Secondary Analysis Traces (Rows 2+)
    plot_secondary_analysis_trends_grouped(
        fig=fig, df=df, scan_variable=scan_variable,
        dataset_column="dataset_name", col_map=layout["col_map"],
        color_map=layout["color_map"], secondary_config=secondary_config,
        sorting_key_column=sorting_key_column,
        trace_name_map=layout["trace_name_map"],
    )

    # --- 3. FINALIZE LAYOUT ---
    logging.info("Calculating and applying unified axes ranges and custom labels...")

    # Set X-Axis Range
    x_col_name = f"{scan_variable}_target"
    x_values = pd.to_numeric(df[x_col_name], errors='coerce').dropna()
    if not x_values.empty:
        x_min, x_max = x_values.min(), x_values.max()
        padding = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.1
        fig.update_xaxes(range=[x_min - padding, x_max + padding])

    # Set Primary Y-Axis Range
    primary_y_range, _ = calculate_global_y_range(df, 'numeric', primary_y_column)

    if primary_y_range[0] is not None:
        # Check if the calculated lower bound is positive.
        # If so, snap it to 0. 
        # Note: If values are negative (infeasible), mdf_range[0] will stay negative.
        if primary_y_range[0] > 0:
            primary_y_range[0] = 0
        fig.update_yaxes(range=primary_y_range, row=1)

    primary_y_label = primary_plot_options.get("y_axis_label", primary_y_column)
    fig.update_yaxes(title_text=primary_y_label, row=1, col=1)

    # Set Secondary Y-Axes Ranges
    for i, panel_config in enumerate(secondary_config):
        target_row = i + 2
        panel_type = panel_config.get("type")

        # 1. Calculate Range (We ignore the manual tick_vals now)
        final_range = [None, None]
        is_log = False

        if panel_type == "FVA":
            items = panel_config.get("reactions_to_show")
            final_range, _ = calculate_global_y_range(df, 'dict_min_max', 'FVA_Results', items_to_plot=items)
        
        elif panel_type == "CVA":
            items = panel_config.get("metabolites_to_show")
            is_log = True
            # We calculate the robust range, but we IGNORE the tick_vals
            final_range, _ = calculate_global_y_range(df, 'dict_min_max', 'CVA_Results', items_to_plot=items, log_scale=True)

        elif panel_type == "Metabolite_Ratios":
            items = panel_config.get("ratios_to_show")
            is_log = True
            final_range, _ = calculate_global_y_range(df, 'dict_min_max', 'Metabolite_Ratios', items_to_plot=items, log_scale=True)

        elif panel_type == "Bottlenecks":
            ranges = []
            if 'Bottleneck_Reactions' in df.columns:
                perturbation_range, _ = calculate_global_y_range(df, 'list_len', 'Bottleneck_Reactions')
                if perturbation_range[0] is not None:
                    ranges.append(perturbation_range)
            if 'Minimal_Bottleneck_Sets' in df.columns:
                minimal_set_range, _ = calculate_global_y_range(df, 'dict_key', 'Minimal_Bottleneck_Sets', items_to_plot='minimal_relaxation_count')
                if minimal_set_range[0] is not None:
                    ranges.append(minimal_set_range)
            if ranges:
                final_range = [min(r[0] for r in ranges), max(r[1] for r in ranges)]

        elif panel_type == "Minimal_Bottleneck_Sets":
            final_range, _ = calculate_global_y_range(df, 'dict_key', 'Minimal_Bottleneck_Sets', items_to_plot='minimal_relaxation_count')

        # 2. Apply to Figure
        if final_range[0] is not None:
            update_dict = {
                'range': final_range,
                'row': target_row,
                'gridcolor': 'white', 
                'gridwidth': 1, 
                # Re-enable minor grids for that nice log-scale look
                'minor_showgrid': True, 
                'minor_gridcolor': 'white', 
                'minor_gridwidth': 0.5 
            }
            
            if is_log:
                update_dict.update({
                    'type': 'log', 
                    'exponentformat': 'power',
                    'dtick': 1, 
                    'ticklabelmode': 'period' 
                })
            
            fig.update_yaxes(**update_dict)

        # 3. Apply Label
        secondary_y_label = panel_config.get("y_axis_label")
        if secondary_y_label is None:
            type_specific_labels = global_axis_labels.get("y_axis_labels", {})
            secondary_y_label = type_specific_labels.get(panel_type)
        if secondary_y_label is None:
            default_y_title = "Count" if panel_type in {"Bottlenecks", "Minimal_Bottleneck_Sets"} else "Range"
            type_name = panel_type.replace("_", " ").replace("Ratios", "Ratio")
            secondary_y_label = f"{type_name} {default_y_title}"
        
        fig.update_yaxes(title_text=secondary_y_label, row=target_row, col=1)

    # --- Final Layout ---
    if title_font_size is None:
        title_font_size = font_size + 8
    if subplot_title_font_size is None:
        subplot_title_font_size = font_size

    if layout_mode == "grouped_overlay":
        plot_width = max(1200, 650 * layout["num_cols"])
    else:
        plot_width = max(800, 400 * layout["num_cols"])

    fig.update_layout(
        title={
            'text': plot_title,
            'font': dict(size=title_font_size) 
        },
        height=300 * layout["num_rows"],
        width=plot_width,
        legend_tracegroupgap=10,
        showlegend=show_legend,
        font=dict(family="Arial, sans-serif", size=font_size) 
    )

    global_x_label = global_axis_labels.get("x_axis_label", f"Target {scan_variable}")
    if layout["num_rows"] > 0:
        if layout_mode in {'side_by_side', 'grouped_overlay'}:
            for c in range(1, layout["num_cols"] + 1):
                fig.update_xaxes(title_text=global_x_label, row=layout["num_rows"], col=c)
        else:
            fig.update_xaxes(title_text=global_x_label, row=layout["num_rows"], col=1)

    fig.update_annotations(font_size=subplot_title_font_size)

    # Run Cleanup   
    _clean_empty_subplots(fig, layout["num_rows"], layout["num_cols"])

    return fig


