"""Parse scan results and calculate colors and axis ranges for MDF figures."""

from typing import List, Tuple
import ast
import logging
import pandas as pd
import numpy as np
import colorsys


def _replace_infeasible_recursively(obj):
    """
    Recursively replaces string 'Infeasible' (case-insensitive, stripped)
    with np.nan within dictionaries and lists.
    """
    if isinstance(obj, dict):
        return {k: _replace_infeasible_recursively(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_infeasible_recursively(elem) for elem in obj]
    elif isinstance(obj, str) and obj.strip().lower() == 'infeasible':
        return np.nan
    return obj

def hex_to_rgb(hex_color):
    """Converts a hex color string to an (R, G, B) tuple (0-1 range).

    Args:
        hex_color (str): The hex color string (e.g., "#RRGGBB").

    Returns:
        Tuple[float, float, float]: A tuple representing the RGB values,
            where each value is in the range [0.0, 1.0].
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Converts an (R, G, B) tuple (0-1 range) to a hex color string.

    Args:
        rgb_color (Tuple[float, float, float]): A tuple of RGB values,
            where each value is in the range [0.0, 1.0].

    Returns:
        str: The corresponding hex color string (e.g., "#RRGGBB").
    """
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)
    )

def parse_dict_from_str(s: str) -> dict:
    """Safely parses a dictionary from its string representation.

    Uses `ast.literal_eval` for security. Handles non-string inputs and
    parsing errors gracefully by returning an empty dictionary.

    Args:
        s: The string to be parsed into a dictionary.

    Returns:
        The parsed dictionary, or an empty dictionary if parsing fails.
    """
    if isinstance(s, dict):
        return _replace_infeasible_recursively(s)
    if not isinstance(s, str) or s in ['N/A', 'nan']:
        return {}
    try:
        parsed_obj = ast.literal_eval(s)
        if isinstance(parsed_obj, dict):
            # Recursively replace 'Infeasible' strings with np.nan
            return _replace_infeasible_recursively(parsed_obj)
        return {}
    except (ValueError, SyntaxError):
        return {}

def parse_list_from_str(s: str) -> list:
    """Safely parses a list from its string representation.

    Uses `ast.literal_eval` for security. Handles non-string inputs and
    parsing errors gracefully by returning an empty list.

    Args:
        s: The string to be parsed into a list.

    Returns:
        The parsed list, or an empty list if parsing fails.
    """
    if isinstance(s, list):
        return _replace_infeasible_recursively(s)
    if not isinstance(s, str) or s in ['N/A', 'nan']:
        return []
    try:
        parsed_obj = ast.literal_eval(s)
        if isinstance(parsed_obj, list):
            # Recursively replace 'Infeasible' strings with np.nan
            return _replace_infeasible_recursively(parsed_obj)
        return []
    except (ValueError, SyntaxError):
        return []


def generate_shades(
    base_color_hex: str,
    num_shades: int,
    min_lightness_factor: float = 0.0,
    max_lightness_factor: float = 0.90,
) -> List[str]:
    """
    Generates a list of hex color strings, starting with a base color
    and getting progressively lighter.

    Args:
        base_color_hex: The starting hex color string (e.g., "#0000FF" for blue).
        num_shades: The total number of shades to generate (including the base).
        min_lightness_factor: The lightness factor for the base color (effectively,
                              how much the original lightness can be scaled down,
                              though typically we start from the base color's lightness).
                              Usually 0 or the base color's actual lightness.
        max_lightness_factor: The target maximum lightness for the lightest shade
                              (e.g., 0.9 means the lightest shade will be 90% light,
                              not pure white which is 1.0). Prevents washing out.

    Returns:
        A list of hex color strings.
    """
    if num_shades <= 0:
        return []
    if num_shades == 1:
        return [base_color_hex]

    r_base, g_base, b_base = hex_to_rgb(base_color_hex)
    h, l_base, s = colorsys.rgb_to_hls(
        r_base, g_base, b_base
    )  # HLS is better for lightness

    shades = []
    for i in range(num_shades):
        # Calculate lightness for the current shade
        # We want to interpolate from l_base to max_lightness_factor
        # For the first shade (i=0), lightness should be l_base
        # For the last shade (i=num_shades-1), lightness should be max_lightness_factor (if l_base is less)
        # or l_base + (max_lightness_factor - l_base) * (i / (num_shades - 1))
        # If l_base is already high, we might not reach max_lightness_factor without distorting
        # Let's make it simpler: increment lightness towards max_lightness_factor

        if num_shades == 1:
            current_l = l_base
        else:
            # Interpolate lightness from l_base up to max_lightness_factor
            # Ensure we don't exceed max_lightness_factor or go below a certain point
            # The effective starting lightness is l_base
            # The effective ending lightness is max_lightness_factor (or l_base if l_base > max_lightness_factor)
            target_l = max(
                l_base, min_lightness_factor
            )  # Start at base lightness or min_lightness
            end_l = max_lightness_factor

            if target_l >= end_l and num_shades > 1:  # If base is already very light
                # In this case, we might want to make it slightly darker or just use base color
                # For simplicity, let's just reduce the step or cap it
                current_l = (
                    l_base - (l_base - end_l) * (i / (num_shades - 1))
                    if l_base > end_l
                    else l_base
                )
            else:
                current_l = target_l + (end_l - target_l) * (i / (num_shades - 1))

        current_l = max(0.0, min(1.0, current_l))  # Clamp lightness between 0 and 1

        r, g, b = colorsys.hls_to_rgb(h, current_l, s)
        shades.append(rgb_to_hex((r, g, b)))

    return shades


def calculate_global_y_range(df, data_type, column_name, items_to_plot=None, log_scale=False):
    """
    Calculates the global min and max y-values for a given metric across the entire DataFrame.
    Uses robust and visually conservative padding logic for both linear and logarithmic scales.

    Args:
        df (pd.DataFrame): The full, combined DataFrame.
        data_type (str): The type of data. One of 'numeric', 'dict_min_max', 'list_len', 'dict_key'.
        column_name (str): The name of the column in the DataFrame to analyze.
        items_to_plot (list, optional): For 'dict_min_max', the specific keys to consider. For 'dict_key', the dictionary key to read.
        log_scale (bool): If True, ensures the calculated range is strictly positive and uses log-appropriate padding.

    Returns:
        list: A list containing [min_value, max_value] for the y-axis range, or [None, None] if no data.
    """
    logging.debug(f"--- Calculating Global Y-Range for '{column_name}' (Type: {data_type}, Log: {log_scale}) ---")
    all_values = []

    # --- 1. Data Extraction ---
    if column_name not in df.columns or df[column_name].isnull().all():
        logging.warning(f"Column '{column_name}' not found or is all empty. Cannot calculate range.")
        return [None, None], None

    if data_type == 'numeric':
        all_values = pd.to_numeric(df[column_name], errors='coerce').dropna().tolist()
        logging.debug(f"  - Extracted {len(all_values)} numeric values.")
    elif data_type == 'list_len':
        counts = df[column_name].apply(lambda x: len(parse_list_from_str(x)))
        all_values = counts.dropna().tolist()
        logging.debug(f"  - Extracted {len(all_values)} list lengths.")
    elif data_type == 'dict_key':
        key = items_to_plot
        if isinstance(items_to_plot, list):
            key = items_to_plot[0] if items_to_plot else None
        if not key:
            logging.debug("  - No key specified for dict_key. Returning None.")
            return [None, None], None
        parsed_series = df[column_name].apply(parse_dict_from_str)
        values = parsed_series.apply(lambda d: d.get(key) if isinstance(d, dict) else None)
        all_values = pd.to_numeric(values, errors='coerce').dropna().tolist()
        logging.debug(f"  - Extracted {len(all_values)} values for key '{key}'.")
    elif data_type == 'dict_min_max':
        if not items_to_plot: 
            logging.debug("  - No items_to_plot specified for dict_min_max. Returning None.")
            return [None, None], None
        logging.debug(f"  - Parsing dictionary column for items: {items_to_plot}")
        parsed_series = df[column_name].apply(parse_dict_from_str)
        expanded_df = pd.DataFrame(parsed_series.tolist(), index=parsed_series.index)
        for item in items_to_plot:
            if item in expanded_df.columns:
                # Drop rows where the item's data is missing
                item_series = expanded_df[item].dropna()
                if not item_series.empty:
                    # Expand the inner {'min': x, 'max': y} dictionaries
                    min_max_df = pd.DataFrame(item_series.tolist(), index=item_series.index)
                    extracted_count = 0
                    if 'min' in min_max_df:
                        mins = pd.to_numeric(min_max_df['min'], errors='coerce').dropna().tolist()
                        all_values.extend(pd.to_numeric(min_max_df['min'], errors='coerce').dropna().tolist())
                        extracted_count += len(mins)
                    if 'max' in min_max_df:
                        maxs = pd.to_numeric(min_max_df['max'], errors='coerce').dropna().tolist()
                        all_values.extend(pd.to_numeric(min_max_df['max'], errors='coerce').dropna().tolist())
                        extracted_count += len(maxs)
                    logging.debug(f"    - Item '{item}': extracted {extracted_count} values (min+max).")
            else:
                logging.debug(f"    - Item '{item}' not found in column data.")

    # --- 2. Pre-Processing ---
    if log_scale:
        all_values = [v for v in all_values if v > 0]

    if not all_values:
        return [None, None], None

    min_val = min(all_values)
    max_val = max(all_values)
    logging.debug(f"  - Raw Data Range: [{min_val:.4e}, {max_val:.4e}]")

    # --- 3. Range & Tick Calculation ---
    if log_scale:
        # LOGIC: "Integer Power Enclosure"
        # We calculate the powers of 10. We use np.isclose to handle cases where
        # a value is effectively an integer power (e.g., 0.01) but float math
        # makes it 0.00999999 or -2.000000004.
        
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        logging.debug(f"  - Log10 Range: [{log_min:.4f}, {log_max:.4f}]")

        # Determine Lower Bound (Start Power)
        # If log_min is close to an integer (e.g. -2.0), use -2.
        # Otherwise, floor it (e.g. -2.1 -> -3).
        if np.isclose(log_min, np.round(log_min)):
            start_power = np.round(log_min)
            logging.debug(f"    -> Min snaps to integer power: {start_power}")
        else:
            start_power = np.floor(log_min)
            logging.debug(f"    -> Min rounded down to: {start_power}")

        # Determine Upper Bound (End Power)
        # If log_max is close to integer (e.g. 2.0), use 2.
        # Otherwise, ceil it (e.g. 1.1 -> 2).
        if np.isclose(log_max, np.round(log_max)):
            end_power = np.round(log_max)
            logging.debug(f"    -> Max snaps to integer power: {end_power}")
        else:
            end_power = np.ceil(log_max)
            logging.debug(f"    -> Max rounded up to: {end_power}")

        # Safety for flat data (min == max)
        if end_power <= start_power:
            end_power = start_power + 1
            logging.debug(f"    -> Range flat, expanding end power to: {end_power}")

        padding = 0.2 
        final_range = [start_power - padding, end_power + padding]
        tick_vals = [10**p for p in np.arange(start_power, end_power + 1)]
        
        logging.debug(f"  - Final Log Range: {final_range}")
        logging.debug(f"  - Explicit Ticks: {tick_vals}")
        
        return final_range, tick_vals
        
    elif data_type == 'list_len':
        final_range = [min(0, min_val - 1), max_val + 1]
        return final_range, None
        
    else: 
        # Linear float range with padding
        data_range = max_val - min_val
        
        if data_range == 0:
            # Case for single-value data (flat line)
            # Add absolute padding based on value magnitude or a default
            padding = abs(max_val) * 0.2 if max_val != 0 else 0.2
            final_range = [min_val - padding, max_val + padding]
            logging.debug(f"  - Zero Range Padding: {padding:.4f} -> Final: {final_range}")
            return final_range, None
        
        # INCREASED PADDING: Changed from 0.05 (5%) to 0.1 (10%)
        padding = data_range * 0.1
        
        final_min = min_val - padding
        final_max = max_val + padding
        
        # Smart Clamping:
        # If the data naturally starts at 0 (or very close), keep the axis at 0
        # so it looks anchored. Otherwise, let it float with padding.
        if min_val >= 0 and final_min < 0: 
             # Only clamp if the dip below zero is relatively small (e.g. just padding)
             # This prevents cutting off data that is truly near zero.
             final_min = 0
        
        if max_val <= 0 and final_max > 0: 
             final_max = 0
            
        final_range = [final_min, final_max]
        logging.debug(f"  - Final Linear Range (with 10% padding): {final_range}")
        return final_range, None
