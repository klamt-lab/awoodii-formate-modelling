"""Plot the MDF and cofactor response to the NADH/NAD lower-bound scan."""

# Import packages
from __future__ import annotations
import logging
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import NullFormatter
from mymodelingpackage.analysis.plotting import generate_mdfscan_custom_plot
from mymodelingpackage.analysis.utils import parse_dict_from_str, parse_list_from_str

# Setup paths and logger
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_FOLDER = "03_NAD_Pool_Scan_with_Bottlenecks_PGCD0"
ANALYSIS_PATTERN = "MDF_Bottlenecks_NADH_NAD_Lower_Bound_Scan_PGCD0"
MODE = "mdf_only"  # Supported values: "mdf_only" or "all".


if __name__ == "__main__":
    if MODE not in {"all", "mdf_only"}:
        raise ValueError(f"MODE must be 'all' or 'mdf_only', received {MODE!r}.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    results_dir = RESULTS_DIR / RESULTS_FOLDER
    candidates = sorted(results_dir.glob(f"**/*{ANALYSIS_PATTERN}*_Formate__Acetate.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {ANALYSIS_PATTERN} scan CSV below {results_dir}, found {len(candidates)}.")

    # Read the selected condition and recover physical ratios from solver values.
    results = pd.read_csv(candidates[0], sep=";")
    results = results[results["Scan_Role"] == "point"].copy()
    if results.empty:
        raise ValueError(f"No VariableScan point rows found in {candidates[0]}.")
    results["nadh_nad_lower_bound"] = np.exp(pd.to_numeric(results["Scan_Axis_Value"], errors="raise"))
    results["nadh_nad_lower_bound_target"] = results["nadh_nad_lower_bound"]
    configuration_name = str(results["configuration_name"].iloc[0])
    match = re.search(r"DStat_(\d+)_(average|t(\d+)h)", configuration_name)
    if match is None:
        raise ValueError(f"Could not parse the D-stat time point from {configuration_name!r}.")
    time_label = "average" if match.group(2) == "average" else f"{match.group(3)} h"
    dataset_name = f"DStat {int(match.group(1)):02d} ({time_label})"
    results["dataset_name"] = dataset_name
    results.sort_values("nadh_nad_lower_bound", inplace=True)

    plot_title = f"{dataset_name}: MDF response to the NADH/NAD lower-bound scan (PGCD0)"
    tick_values = results["nadh_nad_lower_bound"].to_numpy()
    tick_labels = [f"{value:.3g}" for value in tick_values]
    x_limits = np.array([tick_values.min(), tick_values.max()])
    x_padding = (x_limits[1] - x_limits[0]) * 0.03
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match the all-times script's compact MDF view and export formats.
    if MODE == "mdf_only":
        mdf_values = pd.to_numeric(results["MinForce"], errors="raise").to_numpy()
        mdf_figure = go.Figure(go.Scatter(
            x=tick_values,
            y=mdf_values,
            mode="lines+markers",
            name=dataset_name,
            line={"color": "#1a4c72", "width": 2},
            marker={"size": 7},
            hovertemplate="NADH/NAD lower bound: %{x:.4g}<br>MDF: %{y:.4g} kJ mol<sup>-1</sup><extra>%{fullData.name}</extra>",
        ))
        mdf_figure.add_hline(y=0.0, line_dash="dot", line_color="#9b2226")
        mdf_figure.update_xaxes(
            type="linear",
            range=[x_limits[0] - x_padding, x_limits[1] + x_padding],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_labels,
            title="NADH/NAD lower bound [-]",
            title_font={"size": 22},
            tickfont={"size": 18},
        )
        mdf_figure.update_yaxes(title="MDF [kJ mol<sup>-1</sup>]", title_font={"size": 22}, tickfont={"size": 18})
        mdf_figure.update_layout(showlegend=False, height=700, width=1500, template="plotly_white", hovermode="x unified", font={"size": 18})

        static_figure, static_axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
        static_axis.plot(tick_values, mdf_values, color="#1a4c72", marker="o", linewidth=2, label=dataset_name)
        static_axis.axhline(0.0, color="#9b2226", linestyle=":", linewidth=1.5)
        static_axis.set_xlabel("NADH/NAD lower bound [-]", fontsize=22)
        static_axis.set_ylabel("MDF [kJ mol$^{-1}$]", fontsize=22)
        static_axis.set_xscale("linear")
        static_axis.set_xticks(tick_values, tick_labels, rotation=45, ha="right")
        static_axis.tick_params(axis="both", labelsize=18)
        static_axis.grid(True, color="#dddddd", linewidth=0.5)

        output_stem = output_dir / "nadh_nad_lower_bound_scan_mdf_only"
        mdf_figure.write_html(output_stem.with_suffix(".html"), include_plotlyjs=True)
        static_figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
        static_figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(static_figure)
        logging.info("Saved the MDF-only plot for %s with %d points in %s.", dataset_name, len(results), output_dir)
        raise SystemExit(0)

    # Reuse MMP2's multi-panel scan figure for a consistent interactive view.
    datasets = [{"name": dataset_name, "title": dataset_name, "color": "#1a4c72"}]
    secondary_panels = [
        {"type": "CVA", "title": "NAD and NADH concentration ranges", "metabolites_to_show": ["nad_c", "nadh_c"], "y_axis_label": "Concentration [M]"},
        {"type": "Metabolite_Ratios", "title": "NADH/NAD variability", "ratios_to_show": ["NADH/NAD_Ratio"], "y_axis_label": "Ratio [-]"},
        {"type": "Metabolite_Ratios", "title": "NADPH/NADP<sup>+</sup> variability", "ratios_to_show": ["NADPH/NADP_Ratio"], "y_axis_label": "Ratio [-]"},
        {"type": "Metabolite_Ratios", "title": "ATP/ADP variability", "ratios_to_show": ["ATP/ADP_Ratio"], "y_axis_label": "Ratio [-]"},
        {"type": "Metabolite_Ratios", "title": "Fdx<sub>red</sub>/Fdx<sub>ox</sub> variability", "ratios_to_show": ["FDXrd/FDXox_Ratio"], "y_axis_label": "Ratio [-]"},
        {"type": "Bottlenecks", "title": "Reaction bottleneck count", "y_axis_label": "Count"},
    ]
    scan_figure = generate_mdfscan_custom_plot(
        df=results,
        scan_variable="nadh_nad_lower_bound",
        datasets_config=datasets,
        secondary_config=secondary_panels,
        layout_mode="side_by_side",
        plot_title=plot_title,
        primary_plot_options={"y_column": "MinForce", "y_axis_label": "MDF [kJ mol<sup>-1</sup>]"},
        global_axis_labels={"x_axis_label": "NADH/NAD lower bound [-]"},
        show_legend=True,
        vertical_spacing=0.035,
        font_size=14,
        title_font_size=22,
        subplot_title_font_size=16,
    )
    scan_figure.update_xaxes(
        type="linear",
        range=[x_limits[0] - x_padding, x_limits[1] + x_padding],
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
    )
    scan_figure.add_hline(y=0.0, line_dash="dot", line_color="#9b2226", row=1, col=1)
    scan_figure.update_layout(height=1900, width=1250, template="plotly_white", hovermode="x unified")

    scan_figure.write_html(output_dir / "nadh_nad_lower_bound_scan.html", include_plotlyjs=True)

    # Matplotlib supplies deterministic PNG and SVG companions.
    parsed_cva = results["CVA_Results"].apply(parse_dict_from_str)
    parsed_ratios = results["Metabolite_Ratios"].apply(parse_dict_from_str)
    static_panels = [
        (parsed_cva, ["nad_c", "nadh_c"], "NAD and NADH concentration ranges", "Concentration [M]"),
        (parsed_ratios, ["NADH/NAD_Ratio"], "NADH/NAD variability", "Ratio [-]"),
        (parsed_ratios, ["NADPH/NADP_Ratio"], "NADPH/NADP$^+$ variability", "Ratio [-]"),
        (parsed_ratios, ["ATP/ADP_Ratio"], "ATP/ADP variability", "Ratio [-]"),
        (parsed_ratios, ["FDXrd/FDXox_Ratio"], "Fdx$_{red}$/Fdx$_{ox}$ variability", "Ratio [-]"),
    ]
    static_figure, axes = plt.subplots(len(static_panels) + 2, 1, figsize=(13, 21), sharex=True, constrained_layout=True)
    axes[0].plot(tick_values, results["MinForce"], color="#1a4c72", marker="o", linewidth=2)
    axes[0].axhline(0.0, color="#9b2226", linestyle=":", linewidth=1.5)
    axes[0].set_title("MDF")
    axes[0].set_ylabel("kJ mol$^{-1}$")

    line_colors = ["#1a4c72", "#ca6702"]
    for axis, (parsed_values, item_ids, title, y_label) in zip(axes[1:-1], static_panels):
        for item_index, item_id in enumerate(item_ids):
            minimum = np.array([entry.get(item_id, {}).get("min", np.nan) for entry in parsed_values], dtype=float)
            maximum = np.array([entry.get(item_id, {}).get("max", np.nan) for entry in parsed_values], dtype=float)
            color = line_colors[item_index % len(line_colors)]
            axis.plot(tick_values, maximum, color=color, marker="o", linewidth=1.8, label=f"{item_id} max")
            axis.plot(tick_values, minimum, color=color, marker="o", linewidth=1.4, linestyle="--", label=f"{item_id} min")
            axis.fill_between(tick_values, minimum, maximum, color=color, alpha=0.12)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.grid(True, which="both", color="#dddddd", linewidth=0.5)
        axis.legend(loc="best", fontsize=8, ncol=2)

    bottleneck_counts = results["Bottleneck_Reactions"].apply(parse_list_from_str).apply(len).to_numpy()
    axes[-1].step(tick_values, bottleneck_counts, where="mid", color="#ca6702", linewidth=2)
    axes[-1].scatter(tick_values, bottleneck_counts, color="#ca6702", zorder=3)
    axes[-1].set_title("Reaction bottleneck count")
    axes[-1].set_ylabel("Count")
    axes[-1].set_xlabel("NADH/NAD lower bound [-]")
    axes[-1].grid(True, color="#dddddd", linewidth=0.5)
    for axis in axes:
        axis.set_xscale("linear")
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xticks(tick_values, tick_labels, rotation=45, ha="right")
    static_figure.suptitle(plot_title, fontsize=18)
    static_figure.savefig(output_dir / "nadh_nad_lower_bound_scan.svg", bbox_inches="tight")
    static_figure.savefig(output_dir / "nadh_nad_lower_bound_scan.png", dpi=220, bbox_inches="tight")
    plt.close(static_figure)

    logging.info("Saved the multi-panel plot for %s with %d points in %s.", dataset_name, len(results), output_dir)
