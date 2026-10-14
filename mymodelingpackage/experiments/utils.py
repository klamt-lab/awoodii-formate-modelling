import copy
from collections import defaultdict
import csv
import logging
import math
from pathlib import Path
import random
from typing import Any

import yaml

import mymodelingpackage.structures as structs
from mymodelingpackage.solvers.interface import BiologicalSolver
from mymodelingpackage.utilities import get_variant_minimization_targets

logger = logging.getLogger(__name__)


def write_fitted_environment_library(
    campaign_dir: str | Path,
    output_path: str | Path,
    fitted_environment_glob: str = "*/*_fitted_environment.yml",
    summary_filename: str | None = None,
    summary_separator: str = ";",
) -> Path:
    """Combine scenario-wise fitted environments into one component library.

    Current FluxValidation exports already contain the corrected measured fluxes.
    For compatibility with older result folders, this function also reads the
    campaign summary and adds successful validation-objective values that are
    missing from those exports. Redundant ``_<substrate>_to_<product>`` suffixes
    from older environment IDs are removed during aggregation.

    Args:
        campaign_dir: FluxValidation campaign result directory.
        output_path: Destination YAML component-library path.
        fitted_environment_glob: Glob relative to ``campaign_dir`` selecting
            scenario-wise fitted environment files.
        summary_filename: Optional campaign-summary filename. The default is
            ``<campaign directory name>_summary.csv``.
        summary_separator: Field separator used by the campaign summary.

    Returns:
        Path to the generated fitted-environment library.

    Raises:
        FileNotFoundError: If the campaign directory, fitted files, or requested
            summary file do not exist.
        ValueError: If an input YAML has an invalid shape, environment IDs are
            duplicated, or a summary row contains a non-numeric objective value.

    Examples:
        >>> write_fitted_environment_library(
        ...     "Results/01_FluxValidation",
        ...     "config/environments_fitted.yml",
        ... )
        PosixPath('config/environments_fitted.yml')
    """
    campaign_path = Path(campaign_dir)
    destination = Path(output_path)
    if not campaign_path.is_dir():
        raise FileNotFoundError(f"FluxValidation campaign directory does not exist: {campaign_path}")

    fitted_paths = sorted(campaign_path.glob(fitted_environment_glob))
    if not fitted_paths:
        raise FileNotFoundError(
            f"No fitted environment YAML files matched '{fitted_environment_glob}' in {campaign_path}."
        )

    # Older scenario exports do not contain the final validation objective.
    # Index successful summary rows by their task identifier so ATPM or another
    # reaction objective can be restored without project-specific reaction IDs.
    objective_values_by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    resolved_summary_filename = summary_filename or f"{campaign_path.name}_summary.csv"
    summary_path = campaign_path / resolved_summary_filename
    if not summary_path.exists():
        raise FileNotFoundError(f"FluxValidation campaign summary does not exist: {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=summary_separator):
            task_id = str(row.get("Task", "")).strip()
            step_name = str(row.get("Step_Name", "")).strip()
            objective_id = str(row.get("Objective", "")).strip()
            status = str(row.get("Status", "")).strip().lower()
            if (
                not task_id
                or not objective_id
                or step_name == "fit_measurements"
                or status != "optimal"
            ):
                continue
            raw_value = row.get(objective_id) or row.get("Objective_Value")
            try:
                objective_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Summary objective '{objective_id}' for task '{task_id}' is not numeric: {raw_value!r}."
                ) from exc
            if not math.isfinite(objective_value):
                raise ValueError(
                    f"Summary objective '{objective_id}' for task '{task_id}' is not finite: {raw_value!r}."
                )
            objective_values_by_task[task_id].append((objective_id, objective_value))

    # Each scenario file must contribute exactly one environment. Preserve file
    # order so the combined library follows deterministic campaign task order.
    combined_environments: dict[str, dict[str, Any]] = {}
    for fitted_path in fitted_paths:
        with fitted_path.open("r", encoding="utf-8") as handle:
            fitted_document = yaml.safe_load(handle) or {}
        if not isinstance(fitted_document, dict):
            raise ValueError(f"Fitted environment YAML must contain a top-level mapping: {fitted_path}")
        source_environments = fitted_document.get("environments")
        if not isinstance(source_environments, dict) or len(source_environments) != 1:
            raise ValueError(
                f"Fitted environment file must contain exactly one environment: {fitted_path}"
            )
        source_name, environment = next(iter(source_environments.items()))
        if not isinstance(environment, dict):
            raise ValueError(f"Fitted environment '{source_name}' must be a mapping: {fitted_path}")

        substrates = environment.get("substrates") or {}
        products = environment.get("products") or {}
        if (
            not isinstance(substrates, dict)
            or not isinstance(products, dict)
            or len(substrates) != 1
            or len(products) != 1
        ):
            raise ValueError(
                f"Fitted environment '{source_name}' must contain one active substrate and product: {fitted_path}"
            )
        substrate_name = str(next(iter(substrates)))
        product_name = str(next(iter(products)))
        legacy_suffix = f"_{sanitize_token(substrate_name)}_to_{sanitize_token(product_name)}"
        environment_name = (
            str(source_name)[: -len(legacy_suffix)]
            if str(source_name).endswith(legacy_suffix)
            else str(source_name)
        )
        if environment_name in combined_environments:
            raise ValueError(
                f"Duplicate fitted environment ID '{environment_name}' from {fitted_path}."
            )

        # Keep existing scenario entries authoritative and append only missing
        # reaction objectives from legacy campaign summaries.
        existing_reaction_ids: set[str] = set()
        for substrate_entries in substrates.values():
            existing_reaction_ids.update(
                str(entry["id"])
                for entry in substrate_entries
                if isinstance(entry, dict) and entry.get("id")
            )
        existing_reaction_ids.update(
            str(entry["id"])
            for entry in products.values()
            if isinstance(entry, dict) and entry.get("id")
        )
        adjustments = environment.setdefault("substrate_product_adjustments", {}).setdefault(
            substrate_name,
            {},
        ).setdefault(product_name, [])
        existing_reaction_ids.update(
            str(entry["id"])
            for entry in adjustments
            if isinstance(entry, dict) and entry.get("id")
        )
        task_id = fitted_path.stem.removesuffix("_fitted_environment")
        if task_id not in objective_values_by_task:
            raise ValueError(
                f"No successful validation-objective summary row matched fitted environment task '{task_id}'."
            )
        for objective_id, objective_value in objective_values_by_task.get(task_id, []):
            if objective_id in existing_reaction_ids:
                continue
            adjustments.append({"id": objective_id, "lb": objective_value, "ub": objective_value})
            existing_reaction_ids.add(objective_id)

        environment.setdefault("concentrations_file", None)
        environment.setdefault("concentration_adjustments", {})
        combined_environments[environment_name] = environment

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"environments": combined_environments},
            handle,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=None,
        )
    logger.info(
        "Wrote %d fitted environment components from %d scenario files to %s.",
        len(combined_environments),
        len(fitted_paths),
        destination,
    )
    return destination


def is_optimal(status: str | None) -> bool:
    """Checks if the solver status string indicates an optimal solution.

    Args:
        status (str | None): The status string returned by the solver (e.g., 'optimal', 'infeasible').

    Returns:
        bool: True if the status is exactly 'optimal' (case-insensitive), False otherwise.
    """
    return (status or "").lower() == "optimal"


def objective_score(solution: structs.SolverResult) -> float | None:
    if solution is None or not is_optimal(solution.status):
        return None
    if solution.objective_value is None:
        return None
    return float(solution.objective_value)


def dedupe_constraints(constraints: list[list[Any]]) -> list[list[Any]]:
    """Remove repeated constraints while preserving their original order."""
    deduped: list[list[Any]] = []
    seen: set[str] = set()

    for constraint in constraints:
        key = repr(constraint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(constraint)

    return deduped


def resolve_constraint_axis_template(
    axis: structs.TwoVariableAxisConfig,
    candidate_constraints: list[list[Any]],
) -> list[Any]:
    """Resolve the four-part template used by a constraint scan axis.

    A constraint axis can either carry its own ``axis.constraint`` template or
    point at an existing base/point constraint by name via ``axis.target``.
    """
    if axis.constraint is not None:
        template = copy.deepcopy(axis.constraint)
    else:
        template = None
        for constraint in candidate_constraints:
            if isinstance(constraint, list) and len(constraint) >= 3 and str(constraint[2]) == axis.target:
                template = copy.deepcopy(constraint)
                break

    if not (isinstance(template, list) and len(template) == 4):
        raise ValueError(f"Constraint scan axes require either `scan_axis.constraint` as a four-part solver constraint or an existing base/point constraint named '{axis.target}'.")

    if not (isinstance(template, list) and len(template) >= 3 and str(template[2])):
        template[2] = axis.target
    return template


def _replace_none_placeholder(value: Any, target_value: float) -> tuple[Any, int]:
    """Replace ``None`` placeholders in a constraint value template."""
    if isinstance(value, list):
        replacement_count = 0
        replaced = []
        for item in value:
            if item is None:
                replaced.append(float(target_value))
                replacement_count += 1
            else:
                replaced.append(item)
        return replaced, replacement_count
    if value is None:
        return float(target_value), 1
    return value, 0


def build_constraint_axis_point(
    *,
    axis: structs.TwoVariableAxisConfig,
    template_constraint: list[Any],
    target_value: float,
    product_id: str | None,
    biomass_id: str | None,
) -> list[Any]:
    """Build one solver-ready constraint from a constraint-axis scan point."""
    variable_def, template_value, constraint_name_value, sign = copy.deepcopy(template_constraint)
    variable_def = resolve_symbolic_vars(
        variable_def,
        product_id=product_id,
        biomass_id=biomass_id,
    )
    if variable_def is None:
        raise ValueError(f"Could not resolve constraint scan axis variables for '{axis.target}'.")

    sign = str(sign)
    if axis.constraint_value_mode == "set":
        variable_value, replacement_count = _replace_none_placeholder(template_value, float(target_value))
        if replacement_count == 0:
            if sign == "range" and isinstance(template_value, list):
                raise ValueError("Constraint scan axis with sign 'range' and mode 'set' needs a null placeholder in the range value, or use constraint_value_mode 'range_lower', 'range_upper', or 'range_symmetric'.")
            variable_value = float(target_value)
        if sign == "range" and not (isinstance(variable_value, list) and len(variable_value) == 2):
            raise ValueError("Constraint scan axis with sign 'range' must resolve to a two-value range.")
    else:
        if sign != "range":
            raise ValueError(f"Constraint scan axis mode '{axis.constraint_value_mode}' requires a 'range' constraint.")
        if axis.constraint_value_mode == "range_symmetric":
            width = abs(float(target_value))
            variable_value = [-width, width]
        else:
            if not (isinstance(template_value, list) and len(template_value) == 2):
                raise ValueError(f"Constraint scan axis mode '{axis.constraint_value_mode}' requires a two-value range template.")
            lower_bound, upper_bound = template_value
            if axis.constraint_value_mode == "range_lower":
                if upper_bound is None:
                    raise ValueError("Constraint scan axis mode 'range_lower' requires a fixed upper range bound.")
                variable_value = [float(target_value), upper_bound]
            elif axis.constraint_value_mode == "range_upper":
                if lower_bound is None:
                    raise ValueError("Constraint scan axis mode 'range_upper' requires a fixed lower range bound.")
                variable_value = [lower_bound, float(target_value)]
            else:
                raise ValueError(f"Unsupported constraint_value_mode: {axis.constraint_value_mode}")

    return [variable_def, variable_value, str(constraint_name_value or axis.target), sign]


def apply_concentration_axis_point(
    point_concentrations: dict[str, dict[str, float]],
    axis: structs.TwoVariableAxisConfig,
    target_value: float,
) -> None:
    """Apply one concentration-axis scan point to a copied concentration dict."""
    # Concentration axes set one metabolite range.
    # Missing metabolite-specific bounds inherit DEFAULT before the selected side is edited.
    #
    # Example fixed point:
    #   axis.target == "fdxrd_c"
    #   target_value == 1e-4
    #   point_concentrations["fdxrd_c"]["min"] = 1e-4
    #   point_concentrations["fdxrd_c"]["max"] = 1e-4
    default_bounds = point_concentrations.get(axis.target) or point_concentrations.get("DEFAULT") or {"min": 1e-6, "max": 1e-2}
    point_concentrations.setdefault(axis.target, copy.deepcopy(default_bounds))

    # Example max-only point:
    #   point_concentrations["fdxrd_c"]["max"] = 1e-3
    #   the previous min is kept unless it is above the scanned max.
    if axis.bound_type == "max_only":
        point_concentrations[axis.target]["max"] = float(target_value)
        if point_concentrations[axis.target]["min"] > float(target_value):
            point_concentrations[axis.target]["min"] = float(target_value)

    # Example min-only point:
    #   point_concentrations["fdxrd_c"]["min"] = 1e-6
    #   the previous max is kept unless it is below the scanned min.
    elif axis.bound_type == "min_only":
        point_concentrations[axis.target]["min"] = float(target_value)
        if point_concentrations[axis.target]["max"] < float(target_value):
            point_concentrations[axis.target]["max"] = float(target_value)

    # Example fixed point keeps both bounds the same.
    else:
        point_concentrations[axis.target]["min"] = float(target_value)
        point_concentrations[axis.target]["max"] = float(target_value)


def apply_dg0_axis_point(
    point_dG0: dict[str, Any],
    axis: structs.TwoVariableAxisConfig,
    target_value: float,
    *,
    forward_suffix: str,
    reverse_suffix: str,
    axis_name: str = "axis",
) -> None:
    """Apply one dG0-axis scan point to a copied dG0 dict."""
    # dG0 axes use the same modes as global thermo adjustments, but the value
    # comes from this scan coordinate instead of analysis_params.
    # Example add-mode point:
    #   old_dG0 == -18.0
    #   target_value == -2.5
    #   new_dG0 == -20.5
    if axis.target not in point_dG0:
        raise ValueError(f"dG0 {axis_name} target '{axis.target}' is not present in the loaded dG0 dataset.")
    if "dG0" not in point_dG0[axis.target]:
        raise ValueError(f"dG0 {axis_name} target '{axis.target}' has no 'dG0' value.")

    old_dG0 = float(point_dG0[axis.target]["dG0"])

    # Add mode adds the target value to the original dG0, so target -2.5 shifts an original dG0 of -18.0 to -20.5.
    if axis.mode == "add":
        new_dG0 = old_dG0 + float(target_value)

    # Scale mode multiplies the original dG0 by the target value, so 0.5 halves dG0 and 2.0 doubles it.
    elif axis.mode == "scale":
        new_dG0 = old_dG0 * float(target_value)

    # Absolute mode replaces the original dG0 with the target value, so the point directly sets this reaction's dG0.
    else:
        new_dG0 = float(target_value)

    # Set the new dG0 for this point.
    point_dG0[axis.target]["dG0"] = new_dG0
    if axis.pair_policy == "auto_reverse":
        # Split reaction pairs should stay thermodynamic opposites when both ids are present in the dG0 dataset.
        # Example:
        #   R1_F gets -20.5 and R1_R gets 20.5.
        paired_reaction_id = None
        if forward_suffix and axis.target.endswith(forward_suffix):
            paired_reaction_id = axis.target[: -len(forward_suffix)] + reverse_suffix
        elif reverse_suffix and axis.target.endswith(reverse_suffix):
            paired_reaction_id = axis.target[: -len(reverse_suffix)] + forward_suffix
        if paired_reaction_id and paired_reaction_id in point_dG0:
            point_dG0[paired_reaction_id]["dG0"] = -new_dG0


def sanitize_token(value: str) -> str:
    """Sanitize a string so it is safe to embed in filenames."""
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def deduplicate_scenario_filename_label(experiment_name: str, scenario_or_artifact_id: str) -> str:
    """Remove duplicated model/environment context from a scenario filename label.

    Scenario IDs are built as ``model__environment__substrate__product``.
    Campaign experiment names are built as ``model_environment_analysis``. If
    the model/environment prefix from the scenario ID is already represented at
    the start of the experiment name, this function removes that duplicated
    prefix from the filename-visible label.

    Args:
        experiment_name: Experiment name used as the stable filename prefix,
            usually ``model_environment_analysis``.
        scenario_or_artifact_id: Scenario ID or artifact ID that starts with a
            scenario ID, for example
            ``model__environment__substrate__product_step``.

    Returns:
        A sanitized filename label. If the scenario/artifact ID does not match
        the expected shape, or if the experiment name does not already contain
        the same model/environment context, the sanitized original identifier is
        returned unchanged.

    Examples:
        >>> deduplicate_scenario_filename_label(
        ...     "Wildtype_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization",
        ...     "Wildtype__Lac_MeOH_to_Ace_EtOH_noCO2__Lactate_Methanol__Acetate",
        ... )
        'Lactate_Methanol__Acetate'

        Non-matching experiment context is left unchanged:
        >>> deduplicate_scenario_filename_label(
        ...     "Different_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization",
        ...     "Wildtype__Lac_MeOH_to_Ace_EtOH_noCO2__Lactate_Methanol__Acetate",
        ... )
        'Wildtype__Lac_MeOH_to_Ace_EtOH_noCO2__Lactate_Methanol__Acetate'
    """
    # Step 1: Normalize both inputs to the same filename-safe token format used
    # by artifact writers, so comparisons match the actual output filename.
    safe_experiment_name = sanitize_token(str(experiment_name or "").strip("_"))
    safe_identifier = sanitize_token(str(scenario_or_artifact_id or "").strip("_"))

    # Step 2: Split only the first three double-underscore separators. The
    # fourth field may contain product plus step/artifact suffixes.
    parts = safe_identifier.split("__", 3)
    if len(parts) != 4 or not safe_experiment_name:
        return safe_identifier

    # Step 3: Compare the scenario's model/environment prefix against the
    # experiment-name prefix. Only matching prefixes are considered duplicated.
    model_component, environment_component, substrate_name, product_and_suffix = parts
    expected_prefix = f"{model_component}_{environment_component}_"
    if not safe_experiment_name.startswith(expected_prefix):
        return safe_identifier

    # Step 4: Keep the unique scenario tail and any step/artifact suffix.
    return f"{substrate_name}__{product_and_suffix}"


def build_scenario_filename_label_map(
    experiment_name: str,
    scenario_ids: Any,
) -> dict[str, str]:
    """Build collision-safe shortened labels for all scenarios in an experiment.

    This function applies ``deduplicate_scenario_filename_label`` to every
    scenario ID, then detects whether two different scenario IDs would produce
    the same shortened label. Colliding labels fall back to their full sanitized
    scenario IDs so the caller can write files without overwriting another
    scenario's output.

    Args:
        experiment_name: Experiment name used to decide whether model and
            environment context is duplicated in each scenario ID.
        scenario_ids: Iterable of scenario identifiers. Empty values are
            ignored.

    Returns:
        Mapping from each sanitized original scenario ID to the label that
        should be used in filenames. Unique candidates use the shortened label;
        colliding candidates use the original full scenario ID.

    Examples:
        >>> build_scenario_filename_label_map(
        ...     "model_a_env_a_analysis",
        ...     [
        ...         "model_a__env_a__substrate_1__product_1",
        ...         "model_a__env_a__substrate_2__product_1",
        ...     ],
        ... )
        {'model_a__env_a__substrate_1__product_1': 'substrate_1__product_1', 
         'model_a__env_a__substrate_2__product_1': 'substrate_2__product_1'}

        If two inputs would shorten to the same label, both stay fully
        qualified:
        >>> build_scenario_filename_label_map(
        ...     "model_a_env_a_analysis",
        ...     [
        ...         "model_a__env_a__substrate_1__product_1",
        ...         "substrate_1__product_1",
        ...     ],
        ... )
        {'model_a__env_a__substrate_1__product_1': 'model_a__env_a__substrate_1__product_1', 
         'substrate_1__product_1':                 'substrate_1__product_1'}
    """
    # Step 1: Sanitize the original IDs once. The map keys intentionally use
    # these sanitized originals because output writers work with sanitized IDs.
    scenario_id_list = [sanitize_token(str(scenario_id).strip("_")) for scenario_id in scenario_ids if scenario_id]

    # Step 2: Build the optimistic shortened label for each scenario.
    candidate_by_id = {
        scenario_id: deduplicate_scenario_filename_label(experiment_name, scenario_id)
        for scenario_id in scenario_id_list
    }

    # Step 3: Reverse the mapping to find labels shared by multiple scenarios.
    ids_by_candidate: dict[str, set[str]] = defaultdict(set)
    for scenario_id, candidate in candidate_by_id.items():
        ids_by_candidate[candidate].add(scenario_id)

    # Step 4: Use shortened labels only when they are unique; otherwise fall
    # back to the full scenario ID to preserve overwrite safety.
    return {
        scenario_id: scenario_id if len(ids_by_candidate[candidate]) > 1 else candidate
        for scenario_id, candidate in candidate_by_id.items()
    }


def deduplicate_result_filename_stem(experiment_name: str, source_stem: str) -> str:
    """Deduplicate a full result filename stem for summary-table display.

    Result filenames usually have the shape
    ``experiment_name_scenario_or_artifact_label``. This function preserves the
    experiment name once and deduplicates only the scenario/artifact part after
    it. It is also tolerant of stems that do not start with the experiment name.

    Args:
        experiment_name: Experiment name inferred from the result directory or
            calling context.
        source_stem: Filename stem from an individual CSV result file.

    Returns:
        A sanitized filename stem suitable for the summary CSV ``Task`` column.
        When possible, duplicated model/environment context is removed from the
        scenario/artifact part while keeping the experiment name once.

    Examples:
        >>> source_stem = (
        ...     "Wildtype_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization_"
        ...     "Wildtype__Lac_MeOH_to_Ace_EtOH_noCO2__Lactate_Methanol__Acetate"
        ... )
        >>> deduplicate_result_filename_stem(
        ...     "Wildtype_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization",
        ...     source_stem,
        ... )
        'Wildtype_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization_Lactate_Methanol__Acetate'

        Stems without the experiment prefix are treated as scenario/artifact
        labels directly:
        >>> deduplicate_result_filename_stem(
        ...     "Wildtype_Lac_MeOH_to_Ace_EtOH_noCO2_Selected_Characterization",
        ...     "Wildtype__Lac_MeOH_to_Ace_EtOH_noCO2__Lactate_Methanol__Acetate",
        ... )
        'Lactate_Methanol__Acetate'
    """
    # Step 1: Normalize both strings before comparing prefixes.
    safe_experiment_name = sanitize_token(str(experiment_name or "").strip("_"))
    safe_source_stem = sanitize_token(str(source_stem or "").strip("_"))
    if not safe_experiment_name:
        return safe_source_stem

    # Step 2: If the source stem already starts with the experiment name, split
    # that stable prefix from the scenario/artifact remainder.
    experiment_prefix = f"{safe_experiment_name}_"
    if safe_source_stem.startswith(experiment_prefix):
        remainder = safe_source_stem[len(experiment_prefix):]

        # Step 3: Deduplicate only the remainder, then join it back to the
        # experiment name so the visible task label keeps experiment context.
        shortened_remainder = deduplicate_scenario_filename_label(safe_experiment_name, remainder)
        return f"{safe_experiment_name}_{shortened_remainder}"

    # Step 4: If the stem does not include the experiment prefix, treat the
    # entire stem as the candidate scenario/artifact label.
    return deduplicate_scenario_filename_label(safe_experiment_name, safe_source_stem)


def extract_z_vars_from_constraints(
    constraints: list[list[Any]],
    name_filter: str | None = None,
) -> list[str]:
    """Extract z-variable ids referenced inside solver-ready constraints.

    Args:
        constraints: Constraints in solver-ready four-entry list format.
        name_filter: Optional constraint-name filter. When provided, only
            constraints whose third entry matches are inspected.

    Returns:
        list[str]: Sorted z-variable ids found in the constraint list.
    """
    z_vars: set[str] = set()

    for constraint in constraints:
        if not (isinstance(constraint, list) and len(constraint) == 4):
            continue

        variable_def, _, constraint_name, _ = constraint
        if name_filter and constraint_name != name_filter:
            continue

        if isinstance(variable_def, str):
            if variable_def.startswith("z_var_"):
                z_vars.add(variable_def)
        elif isinstance(variable_def, list):
            for item in variable_def:
                if not (isinstance(item, list) and len(item) >= 1):
                    continue
                var_id = item[0]
                if isinstance(var_id, str) and var_id.startswith("z_var_"):
                    z_vars.add(var_id)

    return sorted(z_vars)


def collect_swap_ids(
    scenario: structs.BiologicalScenario,
    minimize_swaps_config: Any,
) -> list[str]:
    """Resolve the z-variables that represent candidate topology swaps.

    This is the reusable, stateless part of swap-target initialization shared by
    multiple experiment classes. Experiment-specific wrappers can then decide
    whether to cache the result, disable swap minimization, or only warn.

    Args:
        scenario: The active biological scenario whose model defines the z-vars.
        minimize_swaps_config: User configuration describing which swap families
            should be considered by the minimization step.

    Returns:
        list[str]: The matching z-variable identifiers. Returns an empty list if
        swap minimization is not configured or no matching variables exist.
    """
    if not minimize_swaps_config:
        return []

    return get_variant_minimization_targets(
        scenario.model,
        scenario.model_conf.model_dump(),
        minimize_swaps_config,
    )


def resolve_symbolic_vars(
    value: Any,
    *,
    product_id: str | None,
    biomass_id: str | None,
) -> Any:
    """Translates human-readable symbolic variables into exact model identifiers.

    This function keeps configuration files clean by allowing users to use generic 
    tags like 'product' or 'biomass'. It handles single strings, lists of strings, 
    and lists of lists containing variables and their multipliers.

    Handles all of these with one function:
    - "product" -> product_id
    - "biomass" -> biomass_id
    - ["product", "ATPM"] -> [product_id, "ATPM"]
    - [["product", 1.0], ["ATPM", -1.0]] -> [[product_id, 1.0], ["ATPM", -1.0]]

    Args:
        value (Any): The symbolic variable name(s) to resolve.
        product_id (str | None): The actual ID of the product in the current scenario.
        biomass_id (str | None): The actual ID of the biomass equation in the current scenario.

    Returns:
        Any: The resolved variable ID(s), maintaining the original input data structure.
    """
    if value is None:
        return None
    # Handle single string resolutions
    if isinstance(value, str):
        if value == "product":
            return product_id
        if value == "biomass":
            return biomass_id
        return value
    # Handle objective-var style lists: ["product", "ATPM"] or [["product", 1.0], ["ATPM", -1.0]]
    if isinstance(value, list):
        # objective-var style: ["product", "ATPM"]
        if all(isinstance(item, (str, type(None))) for item in value):
            resolved_items = []
            for item in value:
                resolved_item = resolve_symbolic_vars(item, product_id=product_id, biomass_id=biomass_id)
                if resolved_item is not None:
                    resolved_items.append(resolved_item)
            return resolved_items

        # linear-expression style: [["product", 1.0], ["ATPM", -1.0]]
        resolved_items = []
        for item in value:
            if isinstance(item, list) and item:
                resolved_head = resolve_symbolic_vars(item[0], product_id=product_id, biomass_id=biomass_id)
                if resolved_head is None:
                    return None
                resolved_items.append([resolved_head, *item[1:]])
            else:
                resolved_items.append(item)
        return resolved_items

    return value


def resolve_shared_constraint(
    raw_constraint: Any,
    *,
    product_id: str | None,
    biomass_id: str | None,
    max_prod: float,
    min_growth: float,
    min_mdf: float,
    fraction_prod: float,
) -> list[list[Any]]:
    if isinstance(raw_constraint, str):
        if raw_constraint == "min_growth":
            return [[biomass_id, min_growth, "min_growth", "geq"]] if biomass_id else []
        if raw_constraint == "min_mdf":
            return [["var_B", min_mdf, "min_mdf", "geq"]]
        if raw_constraint == "min_product":
            if product_id and max_prod:
                return [[product_id, max_prod * fraction_prod, "min_product", "geq"]]
            return []

        logger.debug("Ignoring unknown constraint token: %s", raw_constraint)
        return []

    if isinstance(raw_constraint, list):
        if not raw_constraint:
            return []

        resolved_lhs = resolve_symbolic_vars(raw_constraint[0], product_id=product_id, biomass_id=biomass_id)
        if resolved_lhs is None:
            logger.warning("Skipping unresolved constraint: %s", raw_constraint)
            return []

        return [[resolved_lhs, *raw_constraint[1:]]]

    logger.warning("Unsupported constraint format (%s): %r", type(raw_constraint).__name__, raw_constraint)
    return []


def resolve_constraint_block(
    raw_constraints: list[Any],
    *,
    product_id: str | None,
    biomass_id: str | None,
    max_prod: float,
    min_growth: float,
    min_mdf: float,
    fraction_prod: float,
    objective_var_ids: list[str] | None = None,
) -> list[list[Any]]:
    """Resolve the shared, static constraint formats used by experiments.

    This helper intentionally excludes workflow-specific dictionary formats like
    ``use_result_from`` or ``use_swap_result_from``. Those remain in the
    experiment class that owns the execution flow.

    Args:
        raw_constraints: Raw constraint entries, typically strings like
            ``"min_growth"`` or explicit list constraints.
        product_id: Scenario-specific product reaction id, if available.
        biomass_id: Scenario-specific biomass reaction id.
        max_prod: Precomputed maximum theoretical product yield.
        min_growth: Minimum growth threshold used by the ``min_growth``
            shorthand.
        min_mdf: Minimum MDF threshold used by the ``min_mdf`` shorthand and
            the ``var_B2`` safeguard.
        fraction_prod: Product fraction multiplier used by the ``min_product``
            shorthand.
        objective_var_ids: Active objective ids. When ``"var_B2"`` is present,
            a fallback ``var_B >= min_mdf`` constraint is injected if missing.

    Returns:
        list[list[Any]]: Deduplicated solver-ready constraints.
    """
    resolved: list[list[Any]] = []

    for raw_constraint in raw_constraints or []:
        resolved.extend(
            resolve_shared_constraint(
                raw_constraint,
                product_id=product_id,
                biomass_id=biomass_id,
                max_prod=max_prod,
                min_growth=min_growth,
                min_mdf=min_mdf,
                fraction_prod=fraction_prod,
            )
        )

    if objective_var_ids and "var_B2" in objective_var_ids and not any(
        isinstance(constraint, list) and len(constraint) > 0 and constraint[0] == "var_B"
        for constraint in resolved
    ):
        resolved.append(["var_B", min_mdf, "min_mdf", "geq"])
        logger.debug("Automatically injected 'min_mdf' constraint because 'var_B2' is part of the objective.")

    return dedupe_constraints(resolved)


def set_solver_random_seed(solver: Any, seed: int) -> None:
    """Set the random seed supported by the selected solver adapter.

    Args:
        solver: Solver adapter whose runtime seed should be updated.
        seed: Reproducible integer seed for the next solve.
    """
    solver_config = getattr(solver, "solver_config", None)
    cplex_params = getattr(solver_config, "cplex_params", None)
    if isinstance(cplex_params, dict):
        cplex_params["randomseed"] = seed
    if hasattr(solver, "random_seed"):
        solver.random_seed = seed


def solve_step_robustly(
    *,
    solver: BiologicalSolver,
    objective_var_ids: list[str],
    objective_multipliers: list[float] | None,
    objective_direction: str,
    constraints: list[list[Any]],
    num_retries: int,
    warm_start_solution: dict[str, Any] | None = None,
    logger_: logging.Logger | None = None,
) -> structs.SolverResult:
    """Executes a solver run with an automatic multi-start retry mechanism upon failure.

    Biological network optimizations (especially those involving integer variables like 
    genetic swaps or complex thermodynamic constraints) can sometimes fail to converge 
    or falsely report infeasibility depending on the solver's initial starting point. 
    
    This function attempts an initial solve with a deterministic seed for reproducibility. 
    If that fails, it launches a specified number of retries using different random seeds 
    to help the solver escape local traps or bad initialization states.

    Args:
        solver (Any): The solver interface instance (e.g., BiologicalSolver) used to execute the run.
        objective_var_ids (list[str]): The list of variables to optimize.
        objective_multipliers (list[float] | None): Weights for the objective variables. 
            If None, defaults to 1.0 for all variables.
        objective_direction (str): The direction of optimization ('max' or 'min').
        constraints (list[list[Any]]): A list of constraints to apply to the current solve.
        num_retries (int): Maximum number of retry attempts if the initial solve fails.
        warm_start_solution (dict[str, Any] | None, optional): A previous optimal solution 
            used to initialize the solver, drastically speeding up sequential solves. Defaults to None.
        logger_ (logging.Logger | None, optional): A custom logger instance. If None, 
            the module-level logger is used. Defaults to None.

    Returns:
        Any: The solution object returned by the solver. This will be the optimal solution 
            if successful, or the final failed solution if all retries are exhausted.
    """
    # 1. Setup Logging and Defaults
    active_logger = logger_ or logger

    # 2. Initial Deterministic Solve
    # We use a fixed seed (12345) for the first attempt. This ensures that if the problem 
    # is easily solvable, running the same script twice yields the exact same solver path.
    set_solver_random_seed(solver, 12345)
    solution = solver.solve(
        objective_var_ids=objective_var_ids,
        objective_multipliers=objective_multipliers or [1.0] * len(objective_var_ids),
        objective_direction=objective_direction,
        constraints=constraints,
        warm_start_solution=warm_start_solution,
    )

    # 3. Success Check
    # If the solver found an optimal solution right away, return it immediately.
    if is_optimal(solution.status):
        return solution
    
    # 4. Adaptive Multi-Start Improvement Loop (Error Recovery)
    # If we reach here, the initial solve failed (e.g., status is 'Infeasible', 'TimeLimit', etc.).
    # We log a warning and begin the retry loop.
    active_logger.warning("Initial solve failed (status=%s). Launching up to %d retry attempt(s).", solution.status, num_retries)
    for attempt in range(num_retries):
        # Perturb the seed to give the solver a completely different starting coordinate
        seed = random.randint(10000, 99999)
        set_solver_random_seed(solver, seed)
        active_logger.debug("  -> Retry %d/%d (Seed: %d)...", attempt + 1, num_retries, seed)
        retry_solution = solver.solve(
            objective_var_ids=objective_var_ids,
            objective_multipliers=objective_multipliers or [1.0] * len(objective_var_ids),
            objective_direction=objective_direction,
            constraints=constraints,
            warm_start_solution=warm_start_solution,
        )
        if is_optimal(retry_solution.status):
            active_logger.info("-> SUCCESS: Retry %d/%d succeeded with seed %d.", attempt + 1, num_retries, seed)
            return retry_solution
    
    # 5. Exhaustion
    # If all retries fail, we return the very first failed solution (or you could return 
    # the last one; here we follow your original logic of returning the first `solution`).
    active_logger.error("All %d retry attempts failed. Returning suboptimal solution.", num_retries)
    return solution


def build_quality_lock_constraints(
    solution: structs.SolverResult,
    *,
    objective_var_ids: list[str],
    objective_direction: str,
    prefix: str,
) -> list[list[Any]]:
    """Builds constraints to lock a variable's value based on a previous optimal solution.

    This is crucial for sequential optimizations where you want to maintain the 
    quality of a previously found objective (like Max MDF) while optimizing something 
    else (like minimizing genetic swaps). It adds a small margin (0.1%) to prevent 
    solver infeasibility due to floating-point noise.

    Args:
        solution (Any): The previous optimal solution object containing 'values'.
        objective_var_ids (list[str]): The variables we want to lock down.
        objective_direction (str): The direction of the original optimization ('max' or 'min').
        prefix (str): A string prefix to uniquely name the new constraints.

    Returns:
        list[list[Any]]: A list of new constraints ready to be passed to the solver.
    """
    locks: list[list[Any]] = []
    logger.debug(f"Building quality lock constraints for variables: {objective_var_ids}")
    for var_id in objective_var_ids:
        ideal_value = solution.values.get(var_id)
        if ideal_value is None:
            continue
        # Add a 0.1% margin (or at least 1e-6) to avoid tight floating-point infeasibilities
        margin = max(abs(ideal_value) * 0.001, 1e-6)
        if objective_direction == "max":
            # If we maximized before, we want to ensure the value doesn't drop below the (max - margin)
            locks.append([var_id, ideal_value - margin, f"{prefix}_{var_id}", "geq"])
        else:
            # If we minimized before, we want to ensure the value doesn't exceed the (min + margin)
            locks.append([var_id, ideal_value + margin, f"{prefix}_{var_id}", "leq"])
        logger.debug(f"    -> Locked '{var_id}' near {ideal_value:.5f} (margin: {margin:.5f})")
    return locks


def solution_matches_reference(
    candidate_solution: structs.SolverResult,
    reference_solution: structs.SolverResult,
    objective_var_ids: list[str],
    objective_direction: str,
) -> bool:
    """Validates if a new solution matches the objective quality of a reference solution.

    This acts as a sanity check. For example, if we minimize swaps, does the resulting 
    network topology still yield the same target MDF?

    Args:
        candidate_solution (Any): The new solution to check.
        reference_solution (Any): The ideal reference solution.
        objective_var_ids (list[str]): The variables whose values must match.
        objective_direction (str): The optimization direction ('max' or 'min').

    Returns:
        bool: True if the candidate solution is optimal and matches the reference within tolerance.
    """
    if not is_optimal(candidate_solution.status):
        logger.debug(f"Candidate solution is not optimal (Status: {candidate_solution.status}).")
        return False
    for var_id in objective_var_ids:
        ref_value = reference_solution.values.get(var_id)
        cand_value = candidate_solution.values.get(var_id)
        if ref_value is None or cand_value is None:
            logger.warning(f"Missing values for '{var_id}' in candidate or reference solution.")
            return False
        # Tolerance check allows for minor solver noise (1e-4)
        margin = max(abs(ref_value) * 0.001, 1e-6) + 1e-5
        if objective_direction == "max" and cand_value < ref_value - margin:
            logger.debug(f"Validation failed: Candidate '{var_id}' ({cand_value:.5f}) is worse than reference ({ref_value:.5f}).")
            return False
        if objective_direction == "min" and cand_value > ref_value + margin:
            logger.debug(f"Validation failed: Candidate '{var_id}' ({cand_value:.5f}) is worse than reference ({ref_value:.5f}).")
            return False
    logger.debug("Candidate solution matches reference quality.")
    return True


def solve_with_swap_minimization(
    *,
    solver: BiologicalSolver,
    objective_var_ids: list[str],
    objective_multipliers: list[float] | None,
    objective_direction: str,
    constraints: list[list[Any]],
    swap_ids: list[str],
    num_numerical_retries: int,
    num_swap_attempts: int,
    num_reoptimization_attempts_sandwich: int,
    ideal_solution: structs.SolverResult | None = None,
    quality_prefix: str = "maintain_quality",
    debug_name_filter: str | None = None,
    raise_on_swap_failure: bool = False,
    logger_: logging.Logger | None = None,
) -> tuple[structs.SolverResult, int | str, list[list[Any]]]:
    """Run the shared three-phase swap-minimization sandwich.

    The sandwich proceeds in three phases:
    1. Find or accept an ideal primary-objective solution.
    2. Minimize the active swap variables while locking primary quality.
    3. Re-optimize the original objective under the discovered swap budget.

    Args:
        solver: Solver used for all sandwich phases.
        objective_var_ids: Primary objective variable ids.
        objective_multipliers: Primary objective coefficients aligned with
            ``objective_var_ids``.
        objective_direction: ``"max"`` or ``"min"`` for the primary objective.
        constraints: Base solver-ready constraints before sandwich-specific
            quality locks and swap caps are added.
        swap_ids: Candidate swap variable ids to minimize.
        num_numerical_retries: Retry count used when Phase A needs to solve the
            primary objective itself.
        num_swap_attempts: Number of multi-start attempts for Phase B.
        num_reoptimization_attempts_sandwich: Number of re-optimization
            attempts for Phase C.
        ideal_solution: Optional already-solved Phase A result. When provided,
            the helper skips the initial robust primary solve.
        quality_prefix: Prefix used for the generated quality-lock constraints.
        debug_name_filter: Optional constraint-name filter used to report active
            z-variables in the ideal solution.
        raise_on_swap_failure: If ``True``, raise when no swap-minimized
            solution is found. If ``False``, return the ideal solution instead.
        logger_: Optional logger override.

    Returns:
        tuple[structs.SolverResult, int | str, list[list[Any]]]: Final validated
        solution, realized swap count, and final constraints used for the step
        payload.
    """
    # ------------------------------------------------------------------
    # PHASE 0: SETUP
    # Normalize the logger once so the whole sandwich uses the same sink.
    # ------------------------------------------------------------------
    active_logger = logger_ or logger

    # ------------------------------------------------------------------
    # PHASE 1: IDEAL PRIMARY-OBJECTIVE SOLVE
    # Either accept the already-computed ideal solution from the caller or
    # solve the primary objective robustly here as Step A of the sandwich.
    # ------------------------------------------------------------------
    if ideal_solution is None:
        active_logger.info("    [Sandwich] Step A: Finding ideal solution for %s (%s)...", objective_var_ids, objective_direction)
        ideal_solution = solve_step_robustly(
            solver=solver,
            objective_var_ids=objective_var_ids,
            objective_multipliers=objective_multipliers,
            objective_direction=objective_direction,
            constraints=constraints,
            num_retries=num_numerical_retries,
            logger_=active_logger,
        )
    else:
        active_logger.debug("    [Sandwich] Step A: Using provided ideal solution for %s.", objective_var_ids)

    # Early exit: if the ideal objective itself is not feasible or optimal,
    # there is nothing meaningful to preserve during swap minimization.
    if not is_optimal(ideal_solution.status):
        active_logger.warning("    [Sandwich] Ideal solve failed (%s). Cannot perform swap minimization.", ideal_solution.status)
        return ideal_solution, "N/A", constraints

    # Optional debug reporting: surface how many relevant z-vars are active in
    # the ideal solution before topology minimization changes anything.
    if debug_name_filter:
        debug_zvars = extract_z_vars_from_constraints(constraints, name_filter=debug_name_filter)
        if debug_zvars:
            active_debug = [ var_id for var_id in debug_zvars if ideal_solution.values.get(var_id, 0.0) > 0.5]
            active_logger.info("    [Sandwich] Step A: %s z_vars active in ideal solution: %d", debug_name_filter, len(active_debug))

    # Early exit: if no swap variables were configured or resolved, the ideal
    # solution is already the best answer for this branch.
    if not swap_ids:
        return ideal_solution, "N/A", constraints

    # ------------------------------------------------------------------
    # PHASE 2: TOPOLOGY MINIMIZATION UNDER QUALITY LOCKS
    # Lock the primary-objective quality and minimize the active swap budget.
    # ------------------------------------------------------------------
    quality_constraints = build_quality_lock_constraints(
        ideal_solution,
        objective_var_ids=objective_var_ids,
        objective_direction=objective_direction,
        prefix=quality_prefix,
    )

    best_swap_solution = None
    min_swaps_found: int | None = None
    active_logger.info("    [Sandwich] Step B: Minimizing swaps (%d attempts)...", num_swap_attempts)

    for attempt in range(max(1, num_swap_attempts)):
        seed = 1000 + (attempt * 500)
        set_solver_random_seed(solver, seed)
        active_logger.debug("    -> [Attempt %d] Objective: Minimize %d z_vars (Seed: %d)", attempt + 1, len(swap_ids), seed)

        swap_solution = solver.solve(
            objective_var_ids=swap_ids,
            objective_multipliers=[1.0] * len(swap_ids),
            objective_direction="min",
            constraints=constraints + quality_constraints,
            warm_start_solution=ideal_solution.model_dump() if attempt == 0 else None,
        )

        if not is_optimal(swap_solution.status):
            active_logger.debug("      -> Attempt %d: Swap minimization failed (%s).", attempt + 1, swap_solution.status)
            continue

        current_swaps = int(round(swap_solution.objective_value or 0.0))
        if min_swaps_found is None or current_swaps < min_swaps_found:
            min_swaps_found = current_swaps
            best_swap_solution = swap_solution
            active_logger.debug("      -> Attempt %d: Found new minimum topology: %d swaps.", attempt + 1, current_swaps)

    # Early exit / escalation: callers can choose whether the failure to find a
    # reduced topology should raise or simply fall back to the ideal solution.
    if best_swap_solution is None or min_swaps_found is None:
        if raise_on_swap_failure:
            raise ValueError("No solution was found for min swap optimization")
        active_logger.warning("Swap minimization failed in all %d attempts. Keeping the ideal solution.", num_swap_attempts)
        return ideal_solution, "N/A", constraints

    active_contributors = [var_id for var_id in swap_ids if best_swap_solution.values.get(var_id, 0.0) > 0.5]
    active_logger.info("      -> FOUND MINIMAL TOPOLOGY: %d swaps (out of %d candidates).", min_swaps_found, len(swap_ids))
    active_logger.debug("      -> [Sandwich Step C] Active swap contributors: %s", active_contributors)

    swap_limit_constraint = [[[swap_id, 1.0] for swap_id in swap_ids], min_swaps_found + 0.5, "fixed_swap_count", "leq"]
    final_constraints = constraints + [swap_limit_constraint]

    # ------------------------------------------------------------------
    # PHASE 3: PRIMARY-OBJECTIVE RE-OPTIMIZATION UNDER FIXED SWAP BUDGET
    # Re-solve the original objective while enforcing the minimal topology and
    # validate that the objective quality still matches the ideal solution.
    # ------------------------------------------------------------------
    active_logger.info("    [Sandwich] Step C: Re-optimizing %s with fixed swap count (%d)...", objective_var_ids, min_swaps_found)
    active_logger.debug("    [Sandwich] Step C: Appended constraint to lock swaps: %s", swap_limit_constraint)

    for attempt in range(max(1, num_reoptimization_attempts_sandwich)):
        if attempt > 0:
            seed = random.randint(1000, 50000)
            set_solver_random_seed(solver, seed)
            active_logger.debug("      -> [Attempt %d/%d] Multi-start triggered with seed %d.", attempt + 1, num_reoptimization_attempts_sandwich, seed)
        else:
            active_logger.debug("      -> [Attempt %d/%d] Using best swap solution as warm start.", attempt + 1, num_reoptimization_attempts_sandwich)

        reopt_solution = solver.solve(
            objective_var_ids=objective_var_ids,
            objective_multipliers=objective_multipliers or [1.0] * len(objective_var_ids),
            objective_direction=objective_direction,
            constraints=final_constraints,
            warm_start_solution=best_swap_solution.model_dump() if attempt == 0 else None,
        )

        if solution_matches_reference(
            reopt_solution,
            reference_solution=ideal_solution,
            objective_var_ids=objective_var_ids,
            objective_direction=objective_direction,
        ):
            active_logger.debug("      -> [Attempt %d/%d] SUCCESS: Solution valid.", attempt + 1, num_reoptimization_attempts_sandwich)
            return reopt_solution, min_swaps_found, final_constraints

        active_logger.debug("      -> [Attempt %d/%d] Validation failed or solve not optimal (%s).", attempt + 1, num_reoptimization_attempts_sandwich, reopt_solution.status)

    # If we reach this point, a minimal topology was found but none of the
    # re-optimization attempts could prove that it preserves the ideal quality.
    raise ValueError(f"Could not validate minimal topology for objective variables: {objective_var_ids}.")
