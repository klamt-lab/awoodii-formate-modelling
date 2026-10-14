import os
import csv
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
import json
import re
import yaml
import cobra
from cobra import Model, Reaction
import logging
logger = logging.getLogger(__name__)
import pathlib
import pandas as pd
import copy

from mymodelingpackage.structures import SolverResult

# --- Constants for Key Prefixes and Thresholds ---
Z_VAR_PREFIX = "z_var_"
DG0_VAR_PREFIX = "dG0_"
X_VAR_PREFIX = "x_"
F_VAR_PREFIX = "f_var_"
DEFAULT_Z_THRESHOLD = 0.5 # Consider a reaction active if z_var > 0.5 (i.e., = 1)
DEFAULT_FLUX_THRESHOLD = 1e-6 # Reaction flux must be >= DEFAULT_FLUX_THRESHOLD
VALUES_KEY = "values"


def get_exclusive_reaction_groups(settings: Any) -> list[list[Any]]:
    """Return mutually exclusive reaction groups from typed or raw settings.

    This normalizes the renamed primary key ``exclusive_reaction_groups`` and
    the legacy key ``cofac_alternatives`` to one access path.

    Args:
        settings: A model config object, settings dict, or any object exposing
            one of the supported attributes.

    Returns:
        The configured exclusive reaction groups, or an empty list if absent.
    """
    if settings is None:
        return []
    if hasattr(settings, "exclusive_reaction_groups"):
        return list(getattr(settings, "exclusive_reaction_groups") or [])
    if hasattr(settings, "cofac_alternatives"):
        return list(getattr(settings, "cofac_alternatives") or [])
    if isinstance(settings, dict):
        return list(settings.get("exclusive_reaction_groups") or settings.get("cofac_alternatives") or [])
    return []


def extract_reaction_ids_from_escher_map(map_identifier: str) -> list[str]:
    """Extract reaction base ids from a local Escher map JSON file.

    Args:
        map_identifier: Absolute or relative path to an Escher map JSON file.

    Returns:
        list[str]: Unique reaction identifiers in map order.

    Raises:
        ValueError: If the map format is unsupported or not JSON-backed.
        FileNotFoundError: If the local map file does not exist.
    """
    map_path = Path(map_identifier)
    if map_path.suffix.lower() != ".json":
        raise ValueError(
            "Map-driven FVA currently requires a local Escher JSON map file. "
            f"Received map identifier: {map_identifier}"
        )
    if not map_path.exists():
        raise FileNotFoundError(f"Escher map file not found: {map_identifier}")

    raw_map = json.loads(map_path.read_text(encoding="utf-8"))
    reactions_block = None
    if isinstance(raw_map, list):
        for item in raw_map:
            if isinstance(item, dict) and isinstance(item.get("reactions"), dict):
                reactions_block = item["reactions"]
                break
    elif isinstance(raw_map, dict) and isinstance(raw_map.get("reactions"), dict):
        reactions_block = raw_map["reactions"]

    if not isinstance(reactions_block, dict):
        raise ValueError(f"Unsupported Escher map format in: {map_identifier}")

    reaction_ids: list[str] = []
    for reaction_payload in reactions_block.values():
        if not isinstance(reaction_payload, dict):
            continue
        reaction_id = reaction_payload.get("bigg_id") or reaction_payload.get("id")
        if isinstance(reaction_id, str) and reaction_id:
            reaction_ids.append(reaction_id)

    return list(dict.fromkeys(reaction_ids))


def resolve_model_reactions_from_escher_map(
    map_identifier: str,
    model: cobra.Model,
    *,
    forward_reaction_suffix: str = "_FWD",
    reverse_reaction_suffix: str = "_REV",
) -> list[str]:
    """Resolve Escher map reactions to actual reaction ids present in a model.

    Map reactions are usually stored as reversible base ids. This helper expands
    them to actual model reaction ids, including split forward/reverse variants
    when present.

    Args:
        map_identifier: Local Escher JSON map file.
        model: COBRA model used to resolve map ids to actual model ids.
        forward_reaction_suffix: Forward-direction suffix for split reactions.
        reverse_reaction_suffix: Reverse-direction suffix for split reactions.

    Returns:
        list[str]: Unique model reaction ids derived from the map.
    """
    map_reaction_ids = extract_reaction_ids_from_escher_map(map_identifier)
    model_reaction_ids = {reaction.id for reaction in model.reactions}
    resolved_reaction_ids: list[str] = []

    for base_reaction_id in map_reaction_ids:
        candidates = [base_reaction_id]
        if forward_reaction_suffix:
            candidates.append(f"{base_reaction_id}{forward_reaction_suffix}")
        if reverse_reaction_suffix:
            candidates.append(f"{base_reaction_id}{reverse_reaction_suffix}")

        for candidate in candidates:
            if candidate in model_reaction_ids and candidate not in resolved_reaction_ids:
                resolved_reaction_ids.append(candidate)

    return resolved_reaction_ids


def json_load(path: str) -> Any:
    """Loads a JSON file from a given path.

    Args:
        path (str): The path to the JSON file.

    Returns:
        Any: The loaded data, typically a dictionary or a list.
    """
    with open(path) as f:
        json_data = json.load(f)
    return json_data


# Load and combine the publication campaign YAML components.

def load_settings_components(base_path: str = "config") -> Dict[str, Any]:
    """Load strict component definitions from YAML files in one directory.

    Every file is parsed independently, so YAML anchors must be defined in the
    same file in which they are used. Duplicate YAML keys and duplicate
    component identifiers are rejected instead of being silently overwritten.
    Only the component categories used by the campaign pipeline are returned;
    top-level YAML blocks used solely as local anchor templates are ignored.

    Args:
        base_path: Directory containing YAML component files.

    Returns:
        Component definitions merged by top-level category.

    Raises:
        FileNotFoundError: If ``base_path`` does not exist.
        ValueError: If a YAML file has duplicate keys, a component category is
            not a mapping, or a component identifier is defined more than once.
        yaml.YAMLError: If a YAML file cannot be parsed.
    """
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Component directory not found at: {base_path}")

    sorted_files = sorted(
        filename
        for filename in os.listdir(base_path)
        if filename.endswith((".yml", ".yaml"))
    )
    logging.info("Loading YAML components in order: %s", sorted_files)

    all_components = {}
    component_sources = {}
    for filename in sorted_files:
        filepath = os.path.join(base_path, filename)
        with open(filepath, "r", encoding="utf-8") as config_file:
            yaml_content = config_file.read()

        try:
            root_node = yaml.compose(yaml_content)

            # Inspect the parsed node tree before safe_load because PyYAML
            # otherwise keeps the last occurrence of a duplicate mapping key.
            nodes_to_check = [root_node] if root_node is not None else []
            checked_node_ids = set()
            while nodes_to_check:
                node = nodes_to_check.pop()
                if id(node) in checked_node_ids:
                    continue
                checked_node_ids.add(id(node))
                if isinstance(node, yaml.nodes.MappingNode):
                    keys_in_mapping = set()
                    for key_node, value_node in node.value:
                        if not isinstance(key_node, yaml.nodes.ScalarNode):
                            raise ValueError(f"YAML mapping keys must be scalar values in '{filepath}'.")
                        if key_node.value in keys_in_mapping:
                            line_number = key_node.start_mark.line + 1
                            raise ValueError(
                                f"Duplicate YAML key '{key_node.value}' in '{filepath}' at line {line_number}."
                            )
                        keys_in_mapping.add(key_node.value)
                        nodes_to_check.append(value_node)
                elif isinstance(node, yaml.nodes.SequenceNode):
                    nodes_to_check.extend(node.value)

            # Use a loader instance with PyYAML's standard safe mapping
            # constructor. This avoids relying on process-wide SafeLoader
            # modifications made by other imported modules.
            safe_loader = yaml.SafeLoader(yaml_content)
            safe_loader.yaml_constructors = safe_loader.yaml_constructors.copy()
            safe_loader.yaml_constructors[
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
            ] = yaml.constructor.SafeConstructor.construct_yaml_map
            try:
                file_components = safe_loader.get_single_data()
            finally:
                safe_loader.dispose()
        except yaml.YAMLError as error:
            logging.error("Could not parse component file '%s': %s", filepath, error)
            raise

        if file_components is None:
            continue
        if not isinstance(file_components, dict):
            raise ValueError(f"Component file '{filepath}' must contain a top-level YAML mapping.")

        # Merge only public campaign component categories. Other top-level
        # blocks may define same-file YAML anchors and are intentionally local.
        for category_name in (
            "models",
            "environments",
            "analyses",
            "Campaigns",
            "plotting",
            "plot_profiles",
        ):
            if category_name not in file_components:
                continue
            category_components = file_components[category_name]
            if not isinstance(category_components, dict):
                raise ValueError(f"Component category '{category_name}' in '{filepath}' must be a mapping.")

            merged_category = all_components.setdefault(category_name, {})
            for component_name, component_data in category_components.items():
                component_location = (category_name, component_name)
                if component_name in merged_category:
                    first_source = component_sources[component_location]
                    raise ValueError(
                        f"Component '{component_name}' in category '{category_name}' is defined in both "
                        f"'{first_source}' and '{filepath}'."
                    )
                merged_category[component_name] = component_data
                component_sources[component_location] = filepath

    return all_components


def flatten_list(nested_list):
    """Recursively flattens a nested list structure."""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def assemble_settings(task_definition: List[str], all_components: Dict) -> Dict[str, Any]:
    """Assemble settings from one model, environment, and analysis component.

    Args:
        task_definition: Component identifiers ordered as model, environment,
            and analysis.
        all_components: Component library returned by
            :func:`load_settings_components`.

    Returns:
        Deep-copied and merged settings for the requested task.

    Raises:
        ValueError: If the task is not a component triplet, a required category
            is invalid, or an identifier is absent from its required category.
    """
    if len(task_definition) != 3:
        raise ValueError(
            "Task components must be ordered as [model, environment, analysis]; "
            f"received {len(task_definition)} component(s): {task_definition}."
        )

    final_settings = {}

    for category_name, component_name in zip(
        ("models", "environments", "analyses"),
        task_definition,
    ):
        category_data = all_components.get(category_name)
        if not isinstance(category_data, dict):
            raise ValueError(f"Required component category '{category_name}' is missing or is not a mapping.")
        if component_name not in category_data:
            available_components = ", ".join(sorted(str(name) for name in category_data)) or "none"
            raise ValueError(
                f"Component '{component_name}' was not found in required category '{category_name}'. "
                f"Available components: {available_components}."
            )

        component_data_to_merge = copy.deepcopy(category_data[component_name])
        if not isinstance(component_data_to_merge, dict):
            raise ValueError(f"Component '{component_name}' in category '{category_name}' must be a mapping.")
        for key, value in component_data_to_merge.items():
            if key in final_settings and isinstance(final_settings.get(key), dict) and isinstance(value, dict):
                final_settings[key].update(value)
            else:
                final_settings[key] = value
    
    for top_key, top_value in list(final_settings.items()):
        # Check if a top-level value is a dictionary (like the 'MDFScan' block)
        if isinstance(top_value, dict) and 'x-additional-analysis-params' in top_value:
            # If it contains our special key, perform the merge logic *inside* this block.
            logging.debug(f"Found and processing 'x-additional-analysis-params' inside '{top_key}' block.")
            
            # Pop the additional params from the nested dictionary
            additional_params = top_value.pop('x-additional-analysis-params')
            # Get or create the 'analysis_params' dictionary within the same nested block
            base_params = top_value.setdefault('analysis_params', {})
            
            for key, value in additional_params.items():
                if key in base_params and isinstance(base_params.get(key), list) and isinstance(value, list):
                    # Extend flat list settings when both sides are plain lists.
                    base_params[key].extend(value)
                elif key in base_params and isinstance(base_params.get(key), dict) and isinstance(value, dict):
                    # Merge one level deeper for nested post-analysis config blocks.
                    for nested_key, nested_value in value.items():
                        if (
                            nested_key in base_params[key]
                            and isinstance(base_params[key].get(nested_key), list)
                            and isinstance(nested_value, list)
                        ):
                            base_params[key][nested_key].extend(nested_value)
                        else:
                            base_params[key][nested_key] = nested_value
                else:
                    # Overwrite scalar settings or replace whole blocks.
                    base_params[key] = value

    if 'x-additional-adjustments' in final_settings:
        final_settings.setdefault('model_adjustments', {}).update(final_settings.pop('x-additional-adjustments'))
    
    if 'x-additional-alternatives' in final_settings:
        additional_groups = final_settings.pop('x-additional-alternatives')
        base_groups = final_settings.setdefault('exclusive_reaction_groups', [])
        if isinstance(base_groups, list) and isinstance(additional_groups, list):
            base_groups.extend(additional_groups)
        else:
            final_settings['exclusive_reaction_groups'] = additional_groups

    # Normalize the renamed setting so newer and legacy code paths can both
    # read the assembled settings dictionary safely.
    if 'exclusive_reaction_groups' not in final_settings and 'cofac_alternatives' in final_settings:
        final_settings['exclusive_reaction_groups'] = copy.deepcopy(final_settings['cofac_alternatives'])
    if 'cofac_alternatives' not in final_settings and 'exclusive_reaction_groups' in final_settings:
        final_settings['cofac_alternatives'] = copy.deepcopy(final_settings['exclusive_reaction_groups'])

    if 'x-extend-substrate-product-adjustments' in final_settings:
        # Pop the extension data so it doesn't remain in the final settings
        extension_data = final_settings.pop('x-extend-substrate-product-adjustments')
        
        # Get the base dictionary (create if it doesn't exist)
        base_data = final_settings.setdefault('substrate_product_adjustments', {})

        # Iterate through Substrates (e.g., "H2_CO2", "Formate")
        for substrate, products_dict in extension_data.items():
            if substrate not in base_data:
                base_data[substrate] = {}
            
            # Iterate through Products (e.g., "Butyrate", "Hexanol")
            for product, reaction_list in products_dict.items():
                if product not in base_data[substrate]: base_data[substrate][product] = []
                
                if isinstance(reaction_list, list):
                    # Flatten the list before extending
                    # This allows us to compose lists in YAML like [*part1, *part2]
                    flat_reactions = flatten_list(reaction_list)
                    base_data[substrate][product].extend(flat_reactions)

    if not final_settings:
        raise ValueError(f"Failed to assemble any settings for task: {task_definition}")

    return final_settings


def matches_prefix(s, prefixes):
    """Check if the given string 's' starts with any of the specified prefixes
    followed by an underscore or if the prefix is the entire string.

    Args:
        s (str): The string to check.
        prefixes (List[str]): The list of prefixes to match against.

    Returns:
        bool: True if the string matches any prefix pattern, False otherwise.
    """
    for prefix in prefixes:
        if re.match(rf"^{prefix}(_|$)", s):
            return True
    return False


def count_carbon_atoms(formula):
    """Counts the number of carbon atoms in a chemical formula string.

    Args:
        formula (str): A string representing a chemical formula (e.g., "C6H12O6").

    Returns:
        int: The number of carbon atoms.
    """
    
    # Regular expression to find carbon atoms in the formula
    match = re.search(r"C(\d*)", formula)
    if match:
        # If there's a match but no number, it means there's one carbon atom
        return int(match.group(1)) if match.group(1) else 1
    else:
        # If there's no match, it means there are no carbon atoms
        return 0


def get_max_yield(model: cobra.Model, product_reaction_id: str) -> float:
    """Helper to calculate max theoretical yield on the current model state using standard FBA."""
    try:
        with model:
            model.objective = product_reaction_id
            solution = model.optimize()
            if solution.status == 'optimal':
                return solution.objective_value
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Failed to calculate max yield for {product_reaction_id}: {e}")
        
    return 0.0


def z_var_rea_constraints(sol):
    """Generates constraints to fix binary indicators (z_vars) from a solution.

    Partitions reactions into active and inactive sets based on their z_var
    values and creates constraints to fix this state for subsequent analyses.

    Args:
        sol (Dict): The solution dictionary from the solver.

    Returns:
        Tuple[List, List, List]: A tuple containing:
            - z_var_constraints (List): Constraints to fix binary variables.
            - active_reactions (List): IDs of reactions with z_var > 0.5.
            - inactive_reactions (List): IDs of reactions with z_var < 0.5.
    """
    reactions_dict_values = sol[VALUES_KEY]
    z_var_constraints = []
    active_reactions = []  # The active reactions will be tested if they are bottlenecks
    inactive_reactions = []
    for key, value in reactions_dict_values.items():
        if key.startswith(Z_VAR_PREFIX):
            if value < 0.5: # Variable is considered 'off' (0)
                z_var_constraints.append([key, 0.5, f"z_var_con_{key}", "leq"])
                inactive_reactions.append(key.replace("z_var_", ""))
            elif value > 0.5: # Variable is considered 'on' (1)
                z_var_constraints.append([key, 0.5, f"z_var_con_{key}", "geq"])
                active_reactions.append(key.replace("z_var_", ""))
            else:
                print("z_var_ could not be accounted to 1 or 0 !!!")
                return None
    return z_var_constraints, active_reactions, inactive_reactions


def contains_variant_or_original(input_string: str) -> bool:
    """
    Checks if a string contains 'VARIANT' or 'ORIGINAL' substrings.
    This check is case-sensitive and the substrings can appear anywhere.

    Args:
        input_string: The string to search within.

    Returns:
        True if any 'VARIANT' and 'ORIGINAL' are found, False otherwise.

    Example:
        >>> contains_variant_or_original("RNF_na_VARIANT_1_TCOSA")
        True
        >>> contains_variant_or_original("HDCR_ORIGINAL_H2_TCOSA")
        True
        >>> contains_variant_or_original("PGI_FWD")
        False
    """
    return "VARIANT" in input_string or "ORIGINAL" in input_string


def z_var_variant_and_original_rea_constraints(sol):
    """Generates z_var constraints specifically for TCOSA-style alternative reactions.

    Filters the solution for z_vars associated with 'VARIANT' or 'ORIGINAL'
    reactions and partitions them into active and inactive sets.

    Args:
        sol (Dict): The solution dictionary from the solver.

    Returns:
        Tuple[List, List, List]: A tuple containing:
            - z_var_constraints (List): Constraints to fix binary variables.
            - active_reactions (List): IDs of active alternative reactions.
            - inactive_reactions (List): IDs of inactive alternative reactions.
    """
    if VALUES_KEY not in sol:
        logging.error("z_var_variant_and_original_rea_constraints: Solution dictionary missing 'values' key.")
        return [], [], []
    reactions_dict_values = sol[VALUES_KEY]
    z_var_constraints = []
    active_reactions = []
    inactive_reactions = []

    logging.info("  - Generating constraints to fix active pathway variants...")
    
    for key, value in reactions_dict_values.items():
        if key.startswith("z_var") and contains_variant_or_original(key):  # only the alternative reactions are fixed with there z_var value to get same 'organism' for further testing
            reaction_id = key.replace("z_var_", "")
            
            if value < 0.5:
                z_var_constraints.append([key, 0.5, f"z_var_con_{key}", "leq"])
                inactive_reactions.append(reaction_id)
                logging.debug(f"    -> Locking INACTIVE: {reaction_id}")

            elif value > 0.5:
                z_var_constraints.append([key, 0.5, f"z_var_con_{key}", "geq"])
                active_reactions.append(reaction_id)
                logging.debug(f"    -> Locking ACTIVE:   {reaction_id}")

            else:
                logging.warning(f"    -> Ambiguous z_var value for {key}: {value}. Constraints might be unstable.")
                return None
    logging.info(f"  - Pathway Constraints Generated: {len(active_reactions)} active variants fixed, {len(inactive_reactions)} inactive variants blocked.")
    return z_var_constraints, active_reactions, inactive_reactions


def generate_pathway_constraints(base_solution: Dict, settings: Dict) -> List[List]:
    """Generates constraints to enforce the active metabolic pathway topology.

    This function inspects an optimal `base_solution` to determine which specific
    pathway variants were selected by the solver. It then generates constraints
    to lock this topological choice for subsequent analyses (like FVA or CVA),
    while preserving the physical flexibility of reversible reactions.

    The logic handles two types of pathway definitions from the
    `exclusive_reaction_groups` setting:
    1.  **Explicit Groups:** Lists of distinct reaction IDs (e.g., `["PFOR", "PFL"]`).
    2.  **Implicit Families:** Single-entry lists acting as a base prefix (e.g., `["HDCR"]`).

    The constraint generation strategy is:
    *   **Selected Chemistry:** If a specific variant (e.g., `HDCR_ORIGINAL_...`) is found
        active in the solution, its associated binary variables (`z_var_..._FWD` and
        `z_var_..._REV`) are left **unconstrained**. This allows the solver to flip the
        reaction direction freely during variability analysis, which is physically correct.
    *   **Rejected Chemistry:** All other variants in the same group or family are
        strictly constrained to 0 (`leq 0.5`), preventing the solver from switching to a
        different enzyme to satisfy thermodynamic constraints.

    Args:
        base_solution (Dict): The dictionary representation of the optimal solution
            from which the active pathway topology is derived.
        settings (Dict): The problem configuration dictionary containing the
            `exclusive_reaction_groups` list.

    Returns:
        List[List]: A list of constraint definitions, where each constraint is a list
        of the form `[variable_name, value, constraint_name, direction]`.
        Example: `[['z_var_PFOR_FWD', 0.5, 'fix_path_z_var_PFOR_FWD', 'leq']]`.
    """
    logging.info("  - Generating pathway constraints based on 'exclusive_reaction_groups'...")
    values = base_solution.get("values", {})
    if not values:
        logging.warning("    -> Base solution has no values. No constraints generated.")
        return []
    constraints = []
    groups = get_exclusive_reaction_groups(settings)
    if not groups:
        logging.info("    -> No 'exclusive_reaction_groups' defined in settings. Skipping.")
        return []
    logging.debug(f"    -> Found {len(groups)} alternative groups to process.")
    processed_z_vars = set()
    for group in groups:
        group_str = ", ".join(group)
        logging.debug(f"    -> Processing Group: [{group_str}]")
        # --- 1. Identify the Specific Active Variant ---
        # We need to find the EXACT base ID of the variant that is active.
        # e.g., for Group ["HDCR"], we might find "HDCR_ORIGINAL_H2_TCOSA".
        # e.g., for Group ["PFOR", "PFL"], we might find "PFL".
        active_variant_base_id = None
        # Iterate over all z_vars in solution to find which one belonging to this group is active
        for z_var, val in values.items():
            if val <= 0.5: continue
            # Check if this active z_var belongs to any base_id in the group
            for base_id in group:
                prefix = f"z_var_{base_id}"
                if z_var == prefix or z_var.startswith(f"{prefix}_"):
                    # Found an active variant!
                    # Now extract its specific base ID (remove FWD/REV and z_var_)
                    # e.g. z_var_HDCR_ORIGINAL_H2_TCOSA_FWD -> HDCR_ORIGINAL_H2_TCOSA
                    specific_id = z_var.replace("z_var_", "").replace("_FWD", "").replace("_REV", "")
                    # Special check: Ensure this specific ID actually starts with the group base_id
                    # (This confirms it belongs to the family)
                    if specific_id == base_id or specific_id.startswith(f"{base_id}_"):
                        active_variant_base_id = specific_id
                        logging.debug(f"      -> Found Active Variant: {z_var} (val={val})")
                        logging.debug(f"      -> Selected Chemistry for Group: {active_variant_base_id}")
                        break
            if active_variant_base_id: break
        if active_variant_base_id is None:
            logging.debug("      -> No active chemistry found for this group. All will be locked OFF.")
        # --- 2. Generate Constraints ---
        # Now iterate again to lock everything that matches the group but IS NOT the active variant
        for base_id in group:
            prefix = f"z_var_{base_id}"
            # Find all z_vars belonging to this family/group
            related_vars = [k for k in values.keys() if k == prefix or k.startswith(f"{prefix}_")]
            for z_var in related_vars:
                if z_var in processed_z_vars: continue
                # Re-derive the specific ID for this variable
                current_specific_id = z_var.replace("z_var_", "").replace("_FWD", "").replace("_REV", "")
                # DECISION LOGIC:
                # Is this variable part of the chosen chemistry?
                if active_variant_base_id and current_specific_id == active_variant_base_id:
                    # YES -> Leave Unconstrained (allows FWD/REV flip)
                    logging.debug(f"        -> Leaving Unconstrained (Selected): {z_var}")
                else:
                    # NO -> Lock OFF (Rejected alternative or unused variant)
                    constraints.append([z_var, 0.5, f"fix_path_{z_var}", "leq"])
                    logging.debug(f"        -> Locking OFF (Rejected): {z_var}")
                processed_z_vars.add(z_var)
    logging.info(f"  - Pathway Constraints Generated: {len(constraints)} total constraints.")
    return constraints


def get_variant_minimization_targets(model: cobra.Model, settings: dict, minimize_config: Union[dict, bool] = None) -> list:
    """Identifies reaction variants to target for minimization.

    This function scans the model and configuration to generate a list of binary
    indicator variables (`z_var_*`) representing "alternative" or "non-reference"
    metabolic pathways. These variables are typically used as an objective
    function to minimize the number of modifications required to achieve a
    phenotype (i.e., finding the "Minimal Swap" solution).

    The function accumulates targets from three potential sources:
    1.  **Implicit Alternatives (from `settings`):** Derived from the
        `exclusive_reaction_groups` list.
        *   For groups with multiple entries (e.g., `["Ref", "Alt1", "Alt2"]`),
            the first item is treated as the free reference, and all subsequent
            items are treated as minimization targets.
        *   For groups with a single entry (e.g., `["FamilyBase"]`), the entry
            is treated as a family base name. All reactions in the model starting
            with this base name are added, *excluding* those explicitly marked
            as `_ORIGINAL_`.
    2.  **Explicit List (from `minimize_config`):** Specific reaction IDs provided
        in the `swap_objective_vars` key are added directly.
    3.  **Keywords (from `minimize_config`):** Any reaction ID containing a
        specified substring (e.g., `swap_keyword="Transport"`) is added.

    Args:
        model (cobra.Model): The COBRApy model to search for reactions.
        settings (dict): The main settings dictionary, expected to contain the
            `exclusive_reaction_groups` key defining pathway groups.
        minimize_config (Union[dict, bool], optional): Configuration for manual
            overrides. If a dictionary, it can contain `swap_objective_vars` (list)
            and `swap_keyword` (str or list). If `True`, only the implicit
            strategies from `settings` are used. Defaults to None.

    Returns:
        list: A sorted list of unique variable names (strings) starting with
        `z_var_` that correspond to the identified alternative reactions.
    """
    swap_ids = set()
    logging.info("  - [Swap Detection] Starting collection of swap variables...")
    
    # --- Helper to search model and add to set ---
    def find_and_add_base_id(base_id, source_desc, mode="standard"):
        """
        modes:
        - "standard": Add everything matching base_id (skip duplicates).
        - "family_base": Add matching, exclude _ORIGINAL_.
        - "reference_scan": Add ONLY matching that contains _VARIANT_.
        """
        found_count = 0
        already_present_count = 0
        prefix_pattern = f"{base_id}_"
        
        for r in model.reactions:
            # 1. Match Check
            is_match = (r.id == base_id or r.id.startswith(prefix_pattern))
            if not is_match:
                continue

            # 2. Exclusion/Inclusion Logic
            if mode == "family_base":
                if r.id == base_id or "_ORIGINAL" in r.id:
                    logging.debug(f"      -> [Exclude] Skipping original/base '{r.id}' for family '{base_id}'.")
                    continue
            elif mode == "reference_scan":
                if "_VARIANT" not in r.id:
                    # For references, we ONLY want to penalize variants
                    continue

            # 3. Add to set
            z_name = f"z_var_{r.id}"
            if z_name not in swap_ids:
                swap_ids.add(z_name)
                found_count += 1
                logging.debug(f"      -> [Add] Added '{z_name}' to minimize list.")
            else:
                already_present_count += 1

        # 4. Logging
        total_found = found_count + already_present_count
        if found_count > 0:
            logging.debug(f"    -> [{source_desc}] Added {found_count} new z_vars for base '{base_id}'.")
        elif total_found > 0:
            logging.debug(f"    -> [{source_desc}] '{base_id}' checked: {total_found} matches already in list.")
        else:
            # Only warn if we truly found nothing at all in the model
            logging.warning(f"    -> [{source_desc}] WARNING: No reactions found matching '{base_id}'.")


    # --- Strategy 1: Structural Alternatives (Implicit Swaps from Settings) ---
    alternatives = get_exclusive_reaction_groups(settings)
    if alternatives:
        logging.info(f"    -> [Strategy 1] Checking {len(alternatives)} alternative groups defined in settings...")
        for i, group in enumerate(alternatives):
            if len(group) > 1:
                # Case A: Explicit Alternatives (e.g., ["PFOR", "PFL"])
                reference = group[0]
                swap_candidates = group[1:] # Items 1..N are swaps
                logging.info(f"      -> Group {i+1} {group}: Ref='{reference}' | Candidates={swap_candidates}")
                # 1. Minimize the alternatives (Standard logic)
                for candidate_id in swap_candidates:
                    find_and_add_base_id(candidate_id, "Implicit-Multi")
                # 2. NEW: Check the Reference for Variants!
                # If PFOR is ref, PFOR_VARIANT_NAD should still be minimized.
                find_and_add_base_id(reference, "Implicit-Ref-Scan", mode="reference_scan")
            elif len(group) == 1:
                # Case B: Single Entry Family (e.g., ["HDCR"])
                base_id = group[0]
                logging.info(f"      -> Group {i+1} {group}: Single Family Base. Minimizing variants of '{base_id}'.")
                find_and_add_base_id(base_id, "Implicit-Family", mode="family_base")
    else:
        logging.info("    -> [Strategy 1] No 'exclusive_reaction_groups' found in settings. Skipping implicit detection.")

    # --- Strategy 2 & 3: Config Overrides (Keywords & Explicit List) ---
    # Only proceed if minimize_config is a dictionary (handles 'minimize_swaps: true' case)
    if isinstance(minimize_config, dict):
        logging.info("    -> [Config] 'minimize_swaps' is a dictionary. checking for manual overrides...")
        # Strategy 2: Explicit List
        explicit = minimize_config.get("swap_objective_vars")
        if explicit:
            logging.info(f"    -> [Strategy 2] Processing explicit list of {len(explicit)} base IDs...")
            for item in explicit:
                base_id = item[0] if isinstance(item, list) else item
                find_and_add_base_id(base_id, "Explicit-Config")
        else:
            logging.debug("    -> [Strategy 2] No 'swap_objective_vars' found.")
        # Strategy 3: Keyword Search
        raw_keywords = minimize_config.get("swap_keyword")
        if raw_keywords:
            keywords = [raw_keywords] if isinstance(raw_keywords, str) else raw_keywords
            logging.info(f"    -> [Strategy 3] Searching for reaction keywords: {keywords}")
            for kw in keywords:
                for r in model.reactions:
                    if kw in r.id:
                        z_name = f"z_var_{r.id}"
                        if z_name not in swap_ids:
                            swap_ids.add(z_name)
                            logging.debug(f"      -> [Keyword] Added {z_name}")

    
    result_list = sorted(list(swap_ids))
    logging.info(f"  - [Swap Detection] COMPLETE. Total unique swap variables identified: {len(result_list)}")
    if len(result_list) <= 50:
        logging.info(f"    IDs: {result_list}")
    else:
        logging.info(f"    IDs (first 10): {result_list[:10]} ...")
    return result_list


def generate_cnapy_scenario(reactions_dict: dict, opt_direction="max") -> dict:
    """Reaction dictionary from frunction preprocess_dict() is taken and CNApy scenario files structure is created for loading results in CNApy.

    Args:
    * reactions_dict: dict ~ Dictionary where reaction IDs are keys and flux values are values
    * opt_directions: str ~ "max" or "min" for optimization direction
    """
    fluxes = {key: [value, value] for key, value in reactions_dict.items()}
    structure = {
        "fluxes": fluxes,
        "pinned_reactions": [],
        "description": "",
        "objective_direction": opt_direction,
        "objective_coefficients": {},
        "use_scenario_objective": False,
        "reactions": {},
        "constraints": [],
        "version": 2,
    }
    return structure


def preprocess_solution_for_cnapy(
    raw_solution: SolverResult | dict[str, Any],
    model: cobra.Model,
) -> Dict[str, float]:
    """
    Extracts and cleans flux values from a solver result for CNApy export.

    Thermodynamic solutions are validated strictly: if the solution contains
    any ``z_var_*`` indicator variables, every exported flux must also have its
    matching indicator. Stoichiometric solutions are also supported directly:
    when no ``z_var_*`` variables are present, the function treats the solver
    payload as a plain flux map and exports those reaction values as-is.

    Args:
        solution (Dict[str, Any]): The complete solution dictionary from a solver.

    Returns:
        Dict[str, float]: A dictionary of reaction IDs to their net flux values.

    Example:
        >>> sol = {
        ...     "status": "Optimal",
        ...     "values": {
        ...         "PGI_FWD": 10.0, "PGI_REV": 2.0, "z_var_PGI_FWD": 1, "z_var_PGI_REV": 1,
        ...         "ATPM": 8.3,
        ...         "var_B": 0.5
        ...     }
        ... }
        >>> preprocess_solution(sol)
        {'PGI': 8.0, 'ATPM': 8.3}
    """
    if isinstance(raw_solution, SolverResult):
        status = raw_solution.status
        values = raw_solution.values
    elif isinstance(raw_solution, dict):
        status = raw_solution.get("status")
        values = raw_solution.get("values", {})
    else:
        logging.warning(
            "preprocess_solution_for_cnapy received unsupported solution type: %s",
            type(raw_solution).__name__,
        )
        return {}

    if str(status or "").lower() != "optimal":
        logging.warning("preprocess_solution_for_cnapy received a non-optimal or empty solution.")
        return {}
    if not values:
        return {}
        
    flux_dict = {}
    processed_bases = set()
    model_reaction_ids = {rxn.id for rxn in model.reactions}
    patterns = ["dG0_", "z_var_", "f_var_", "x_", "var_z_cluster_", "var_B"]
    has_indicator_variables = any(key.startswith("z_var_") for key in values)

    if not has_indicator_variables:
        logging.debug("preprocess_solution_for_cnapy detected a stoichiometric-style solution without z_var indicators.")
        for key, value in values.items():
            if any(key.startswith(pattern) for pattern in patterns):
                continue
            if key not in model_reaction_ids:
                continue

            base_key = key.replace("_FWD", "").replace("_REV", "")
            if base_key in processed_bases:
                continue

            fwd_key = f"{base_key}_FWD"
            rev_key = f"{base_key}_REV"
            is_split_reaction = fwd_key in values or rev_key in values

            if is_split_reaction:
                flux_dict[base_key] = float(values.get(fwd_key, 0.0)) - float(values.get(rev_key, 0.0))
                processed_bases.add(base_key)
                continue

            flux_dict[key] = float(value)
            processed_bases.add(key)

        return flux_dict

    # First, iterate through all potential flux variables to build the flux_dict
    for key in values:
        if any(key.startswith(pattern) for pattern in patterns):
            continue
        if key not in model_reaction_ids:
            continue
        base_key = key.replace("_FWD", "").replace("_REV", "")
        if base_key in processed_bases:
            continue

        # --- This section handles both split and single reactions ---
        fwd_key = f"{base_key}_FWD"
        rev_key = f"{base_key}_REV"

        # Check if it's a split reaction pair
        is_split_reaction = fwd_key in values or rev_key in values
        if is_split_reaction:
            z_fwd_key = f"z_var_{fwd_key}"
            z_rev_key = f"z_var_{rev_key}"
            # --- Enforce z_var existence ---
            if z_fwd_key not in values or z_rev_key not in values:
                if fwd_key in values and z_fwd_key not in values:
                    raise KeyError(f"Inconsistent solution: Flux '{fwd_key}' exists but its indicator '{z_fwd_key}' is missing.")
                if rev_key in values and z_rev_key not in values:
                     raise KeyError(f"Inconsistent solution: Flux '{rev_key}' exists but its indicator '{z_rev_key}' is missing.")
            z_fwd_active = values.get(z_fwd_key, 0.0) > 0.5
            z_rev_active = values.get(z_rev_key, 0.0) > 0.5
            if z_fwd_active or z_rev_active:
                # Export the realized net flux from the active directions.
                # Some thermo solutions leave both indicators active, so choosing
                # only the forward branch can incorrectly collapse a pure reverse
                # flux to zero in the CNApy export.
                flux_dict[base_key] = (
                    (float(values.get(fwd_key, 0.0)) if z_fwd_active else 0.0)
                    - (float(values.get(rev_key, 0.0)) if z_rev_active else 0.0)
                )
                if abs(float(values.get(fwd_key, 0.0))) > DEFAULT_FLUX_THRESHOLD and abs(float(values.get(rev_key, 0.0))) > DEFAULT_FLUX_THRESHOLD: 
                    raise ValueError(f"Inconsistent split reaction '{base_key}': both '{fwd_key}' and '{rev_key}' carry non-zero flux. Exactly one direction must be zero.")
            else:
                # If neither is active, the flux is zero.
                flux_dict[base_key] = 0.0
            processed_bases.add(base_key)
        
        else: # It's a single, unsplit reaction
            z_key = f"z_var_{key}"
            # --- Enforce z_var existence ---
            if z_key not in values:
                raise KeyError(f"Inconsistent solution: Flux '{key}' exists but its indicator '{z_key}' is missing.")
            if values.get(z_key, 0.0) > 0.5:
                flux_dict[key] = values.get(key, 0.0)
            else:
                flux_dict[key] = 0.0
            processed_bases.add(key)
            
    return flux_dict


def generate_summary_csv(campaign_results_dir: str) -> None:
    """
    Scans a campaign directory recursively for all individual .csv result files,
    combines them into a single summary DataFrame, and saves it as a new CSV file.

    It intelligently adds a 'Source_File' column to the summary, containing the
    name of the original CSV file for each row, which is crucial for tracing
    the origin of each result.

    Args:
        campaign_results_dir (str): The path to the campaign folder containing the
                                    per-experiment result subdirectories and CSVs.
    """
    logging.info(f"######################################################################")
    logging.info(f"##  GENERATING CAMPAIGN SUMMARY")
    logging.info(f"######################################################################")
    
    campaign_path = pathlib.Path(campaign_results_dir)
    if not campaign_path.is_dir():
        logging.error(f"Error: The specified path is not a valid directory: {campaign_path}")
        return

    # --- 1. File Discovery ---
    # Campaign outputs are stored in numbered experiment subdirectories
    # (`000_*`, `001_*`, ...), so summary generation must scan recursively
    # instead of only looking at the campaign root directory.
    # Exclude:
    # - existing summary files on reruns
    # - editor/LibreOffice lock files such as `.~lock.foo.csv#`
    csv_files = sorted([f for f in campaign_path.rglob("*.csv") if not f.name.endswith("_summary.csv") and not f.name.startswith(".~lock")])
    
    if not csv_files:
        logging.warning(f"No individual CSV result files found in {campaign_path}. No summary generated.")
        return

    logging.info(f"Found {len(csv_files)} CSV files to combine.")

    from mymodelingpackage.experiments.utils import deduplicate_result_filename_stem

    def _experiment_name_from_result_path(csv_file: pathlib.Path) -> str:
        if csv_file.parent == campaign_path:
            return ""
        parent_name = csv_file.parent.name
        match = re.match(r"^\d+_(.+)$", parent_name)
        return match.group(1) if match else parent_name

    task_candidates = {
        csv_file: deduplicate_result_filename_stem(
            _experiment_name_from_result_path(csv_file),
            csv_file.stem,
        )
        for csv_file in csv_files
    }
    stems_by_candidate: dict[str, set[str]] = {}
    for csv_file, task_candidate in task_candidates.items():
        stems_by_candidate.setdefault(task_candidate, set()).add(csv_file.stem)
    task_names = {
        csv_file: (
            csv_file.stem
            if len(stems_by_candidate[task_candidates[csv_file]]) > 1
            else task_candidates[csv_file]
        )
        for csv_file in csv_files
    }

    # --- 2. Data Loading and Augmentation ---
    all_dataframes = []
    for csv_file in csv_files:
        try:
            # Load the individual CSV file
            df = pd.read_csv(csv_file, delimiter=';')
            
            if df.empty:
                logging.warning(f"  -> Skipping empty file: {csv_file.name}")
                continue

            # .stem is a pathlib feature that gets the filename without the extension.
            source_name = csv_file.stem
            df.insert(0, 'Task', task_names[csv_file])
            
            all_dataframes.append(df)
            logging.debug(f"  -> Processed: {csv_file.name} ({len(df)} rows)")
            
        except Exception as e:
            logging.error(f"  -> Failed to process file {csv_file.name}: {e}")

    if not all_dataframes:
        logging.error("No valid data could be loaded from any CSV files. Summary not created.")
        return

    # --- 3. Concatenation ---
    # Combine all the individual DataFrames into one large master DataFrame
    summary_df = pd.concat(all_dataframes, ignore_index=True)
    
    # --- 4. Save Output ---
    output_path = campaign_path / f"{campaign_path.name}_summary.csv"
    try:
        summary_df.to_csv(output_path, index=False, sep=';')
        logging.info(f"✅ Campaign summary with {len(summary_df)} total rows saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save summary file: {e}")


def is_optimal(result: SolverResult) -> bool:
    """Checks if the solver found an optimal solution."""
    return result.status.lower() == "optimal"

def get_value(result: SolverResult, var_id: str, default: float | None = None) -> float | None:
    """Safely retrieves a variable's value."""
    return result.values.get(var_id, default)

