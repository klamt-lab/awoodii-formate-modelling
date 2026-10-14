"""
Base Experiment module for the new modular batch framework.

This module provides the BaseExperiment class, which encapsulates common
functionality like setting up output directories, saving batch results to 
JSON/CSV/Excel, and triggering post-processing artifacts (Escher maps, CNApy files).

It is entirely decoupled from the mathematical solvers and acts purely 
as an I/O handler and exporter for campaign-level data.
"""

import copy
import gzip
import hashlib
import json
import logging
import os
from math import exp
from typing import Any

import cobra
import pandas as pd
import yaml
from cobrak.constants import (
    ALPHA_VAR_PREFIX,
    DF_VAR_PREFIX,
    ENZYME_VAR_PREFIX,
    GAMMA_VAR_PREFIX,
    IOTA_VAR_PREFIX,
    KAPPA_VAR_PREFIX,
    PROT_POOL_REAC_NAME,
)
from cobrak.spreadsheet_functionality import OptimizationDataset, create_cobrak_spreadsheet
from openpyxl.styles import Alignment, Font, PatternFill

import mymodelingpackage.structures as structs
from mymodelingpackage.experiments.utils import (
    build_scenario_filename_label_map,
    deduplicate_scenario_filename_label,
)
from mymodelingpackage.utilities import (
    generate_cnapy_scenario,
    get_max_yield,
    preprocess_solution_for_cnapy,
)

logger = logging.getLogger(__name__)
MAX_ARTIFACT_FILENAME_BYTES = 240
CNAPY_TOLERANT_EXPORT_EPSILON = 1e-5
CNAPY_TOLERANT_ZERO_THRESHOLD = 1e-6


class BaseExperiment:
    """
    Provides common I/O, logging, and export functionalities for batch experiments.
    
    Subclasses should inherit this class, perform their logic in a `run()` 
    method, and store their final list of payloads in `self.aggregate_results`.
    """

    def __init__(self, experiment_def: structs.ExperimentDefinition, output_dir: str = "./Results"):
        self.experiment_def = experiment_def
        self.output_dir = output_dir
        self.aggregate_results: list[dict[str, Any]] = []
        scenario_ids = [
            scenario.scenario_id
            for scenario in getattr(self.experiment_def, "biological_scenarios", [])
            if getattr(scenario, "scenario_id", None)
        ]
        self._scenario_filename_labels = build_scenario_filename_label_map(
            self.experiment_def.experiment_name,
            scenario_ids,
        )
        self._artifact_path_registry: dict[str, str] = {}
        self._excel_fluxes_by_scenario: dict[str, dict[str, float]] = {}
        self._cobrak_excel_datasets: dict[str, OptimizationDataset] = {}
        
        os.makedirs(self.output_dir, exist_ok=True)

    def _scenario_filename_label(
        self,
        scenario_id: str,
        scenario_label_map: dict[str, str] | None = None,
    ) -> str:
        """Return the shortened filename label for a scenario id."""
        safe_scenario_id = str(scenario_id).strip("_").replace("/", "_").replace("\\", "_")
        if scenario_label_map and safe_scenario_id in scenario_label_map:
            return scenario_label_map[safe_scenario_id]
        if safe_scenario_id in self._scenario_filename_labels:
            return self._scenario_filename_labels[safe_scenario_id]
        return deduplicate_scenario_filename_label(self.experiment_def.experiment_name, safe_scenario_id)

    def _artifact_path_for_parts(self, normalized_parts: list[str], extension: str) -> str:
        """Build an artifact path with a filename short enough for common filesystems.

        Short names stay unchanged. Oversized names are reduced to a readable
        prefix plus a stable hash and the final artifact suffix.
        """
        stem = "_".join(normalized_parts)
        extension = extension.lstrip(".")
        filename = f"{stem}.{extension}"
        if len(filename.encode("utf-8")) <= MAX_ARTIFACT_FILENAME_BYTES:
            return os.path.join(self.output_dir, filename)

        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        prefix = normalized_parts[0] if normalized_parts else "artifact"
        scenario_fragment = normalized_parts[1] if len(normalized_parts) > 2 else ""
        suffix = normalized_parts[-1] if len(normalized_parts) > 1 else "artifact"

        prefix = prefix[:48]
        scenario_fragment = scenario_fragment[:48]
        suffix = suffix[:48]
        compact_parts = [part for part in (prefix, scenario_fragment, digest, suffix) if part]
        compact_stem = "_".join(compact_parts)
        compact_filename = f"{compact_stem}.{extension}"

        if len(compact_filename.encode("utf-8")) > MAX_ARTIFACT_FILENAME_BYTES:
            budget = MAX_ARTIFACT_FILENAME_BYTES - len(f"_{digest}_.{extension}".encode("utf-8"))
            prefix_budget = max(16, budget // 2)
            suffix_budget = max(16, budget - prefix_budget)
            compact_stem = f"{prefix[:prefix_budget]}_{digest}_{suffix[:suffix_budget]}"
            compact_filename = f"{compact_stem}.{extension}"

        logger.warning(f"[{self.experiment_def.experiment_name}] Shortened oversized artifact filename to {compact_filename}")
        return os.path.join(self.output_dir, compact_filename)

    def _reserve_artifact_path(
        self,
        filepath: str,
        source_identity: str,
        fallback_parts: list[str],
        extension: str,
    ) -> str:
        """Reserve one artifact path and fall back to full ids on collisions."""
        registry_key = os.path.abspath(filepath)
        existing_source = self._artifact_path_registry.get(registry_key)
        if existing_source is None:
            self._artifact_path_registry[registry_key] = source_identity
            return filepath

        if existing_source == source_identity:
            raise RuntimeError(f"Artifact path requested more than once in one run: {filepath}")

        fallback_filepath = self._artifact_path_for_parts(fallback_parts, extension)
        fallback_key = os.path.abspath(fallback_filepath)
        if fallback_key == registry_key:
            raise RuntimeError(
                f"Artifact filename collision between '{existing_source}' and '{source_identity}' at {filepath}"
            )

        fallback_existing_source = self._artifact_path_registry.get(fallback_key)
        if fallback_existing_source is not None:
            raise RuntimeError(
                f"Artifact fallback filename collision between '{fallback_existing_source}' and "
                f"'{source_identity}' at {fallback_filepath}"
            )

        self._artifact_path_registry[fallback_key] = source_identity
        logger.warning(
            "[%s] Artifact filename collision for '%s'; using full source id in %s",
            self.experiment_def.experiment_name,
            source_identity,
            fallback_filepath,
        )
        return fallback_filepath

    def _build_artifact_filepath(
        self,
        *parts: str,
        extension: str,
        source_id: str | None = None,
        deduplicate: bool = True,
    ) -> str:
        """Build a collision-checked artifact path using deduplicated labels."""
        normalized_parts = [str(part).strip("_").replace("/", "_").replace("\\", "_") for part in parts if part]
        filename_parts = list(normalized_parts)
        if (
            deduplicate
            and len(filename_parts) >= 2
            and filename_parts[0] == self.experiment_def.experiment_name
        ):
            filename_parts[1] = self._scenario_filename_label(filename_parts[1])

        source_identity = source_id or "_".join(normalized_parts)
        filepath = self._artifact_path_for_parts(filename_parts, extension)
        return self._reserve_artifact_path(filepath, source_identity, normalized_parts, extension)

    def save_to_json(self) -> None:
        """Dumps the raw self.aggregate_results dictionary to a formatted JSON file."""
        if not self.aggregate_results:
            logger.warning(f"[{self.experiment_def.experiment_name}] No results to save to JSON.")
            return

        filepath = os.path.join(self.output_dir, f"{self.experiment_def.experiment_name}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.aggregate_results, f, indent=4)
            logger.info(f"[{self.experiment_def.experiment_name}] Saved complete batch results to {filepath}")
        except Exception as e:
            logger.error(f"[{self.experiment_def.experiment_name}] Failed to save JSON: {e}")

    def save_to_excel(
        self,
        model: cobra.Model | None = None,
        raw_solution: structs.SolverResult | dict[str, Any] | None = None,
        scenario_id: str | None = None,
        workbook_id: str | None = None,
    ) -> str | None:
        """Write a solution into the experiment Summary and Fluxes workbook.

        Args:
            model: Scenario model used to normalize a supplied solution.
            raw_solution: Optional solver result to add to the Fluxes sheet.
            scenario_id: Column label for the supplied solution.
            workbook_id: Optional identifier appended to both workbook names.
                Parallel experiments use this to prevent workers from writing
                to the same files.

        Returns:
            Workbook path, or ``None`` when ``excel_output`` is disabled or no
            experiment results are available.

        Raises:
            ValueError: If a solution is supplied without its model and
                scenario identifier, or if the identifier is duplicated.
        """
        analysis_params = getattr(self.experiment_def.analysis_config, "analysis_params", None)
        if not getattr(analysis_params, "excel_output", False):
            return None

        # Add the current successful solution directly, matching the other
        # optional artifact actions called from each experiment workflow.
        if raw_solution is not None:
            if model is None or scenario_id is None:
                raise ValueError("Excel solution export requires both model and scenario_id.")
            column_label = str(scenario_id).replace("__", "\n")
            if column_label in self._excel_fluxes_by_scenario:
                raise ValueError(f"Duplicate Excel flux-solution identifier: {scenario_id}")
            solution_fluxes = preprocess_solution_for_cnapy(raw_solution, model)
            if not solution_fluxes:
                logger.warning("[%s] No reaction fluxes were available for the Excel workbook.", scenario_id)
                return None
            self._excel_fluxes_by_scenario[column_label] = {
                str(reaction_id): float(value)
                for reaction_id, value in solution_fluxes.items()
            }

        if not self.aggregate_results and not self._excel_fluxes_by_scenario:
            logger.warning("[%s] No results available for Excel export.", self.experiment_def.experiment_name)
            return None

        # Reuse the same standardized rows as CSV output so both formats stay
        # aligned across every BaseExperiment subclass.
        summary_frames = []
        for run_payload in self.aggregate_results:
            rows = self._build_csv_rows(run_payload)
            dataframe = self._build_csv_dataframe(rows)
            dataframe.insert(0, "scenario_id", run_payload.get("scenario_id", "Unknown_Scenario"))
            summary_frames.append(dataframe)
        summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        fluxes = (
            pd.DataFrame(self._excel_fluxes_by_scenario).sort_index()
            if self._excel_fluxes_by_scenario
            else None
        )
        if fluxes is not None:
            fluxes.index.name = "Reaction ID"

        workbook_stem = self.experiment_def.experiment_name
        if workbook_id:
            workbook_stem = f"{workbook_stem}_{self._scenario_filename_label(workbook_id)}"
        output_path = os.path.join(self.output_dir, f"{workbook_stem}.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if not summary.empty:
                summary.to_excel(writer, sheet_name="Summary", index=False)
            if fluxes is not None and not fluxes.empty:
                fluxes.to_excel(writer, sheet_name="Fluxes")

            # Freeze and format the headers so large result sheets remain easy
            # to navigate while preserving the full numeric flux precision.
            for worksheet in writer.book.worksheets:
                worksheet.freeze_panes = "B2" if worksheet.title == "Fluxes" else "A2"
                worksheet.auto_filter.ref = worksheet.dimensions
                worksheet.row_dimensions[1].height = 54 if worksheet.title == "Fluxes" else 30
                for cell in worksheet[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for column_cells in worksheet.iter_cols():
                    header = str(column_cells[0].value or "")
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(len(header) + 2, 14), 42)
                    if worksheet.title == "Fluxes" and column_cells[0].column > 1:
                        for cell in column_cells[1:]:
                            cell.number_format = "0.###############"

        sheet_names = (["Summary"] if not summary.empty else []) + (["Fluxes"] if fluxes is not None else [])
        logger.info("[%s] Generated Excel workbook with sheets=%s at '%s'.", self.experiment_def.experiment_name, sheet_names, output_path)

        # Optional COBRA-k comparison workbook. Keep the existing MMP2 workbook
        # unchanged and write COBRA-k's own spreadsheet next to it.
        if isinstance(raw_solution, structs.SolverResult) and raw_solution.native_result and hasattr(getattr(self, "solver", None), "_prepare_cobrak_model"):
            primary_result = raw_solution.native_result.get("primary")
            if isinstance(primary_result, dict) and primary_result:
                cobrak_output_path = os.path.join(self.output_dir, f"{workbook_stem}_cobrak.xlsx")

                # Show every native reaction-level result that is available. Checking
                # the result keys keeps LP-only exports valid when NLP variables do not exist.
                variable_ids = primary_result.keys()
                has_df = any(variable_id.startswith(DF_VAR_PREFIX) and primary_result.get(variable_id.removeprefix(DF_VAR_PREFIX), 0.0) > 1e-6 for variable_id in variable_ids)
                has_enzyme = any(variable_id.startswith(ENZYME_VAR_PREFIX) for variable_id in variable_ids)
                has_kappa = any(variable_id.startswith(KAPPA_VAR_PREFIX) and primary_result.get(variable_id.removeprefix(KAPPA_VAR_PREFIX), 0.0) > 1e-6 for variable_id in variable_ids)
                has_gamma = any(variable_id.startswith(GAMMA_VAR_PREFIX) and primary_result.get(variable_id.removeprefix(GAMMA_VAR_PREFIX), 0.0) > 1e-6 for variable_id in variable_ids)
                has_iota = any(variable_id.startswith(IOTA_VAR_PREFIX) and primary_result.get(variable_id.removeprefix(IOTA_VAR_PREFIX), 0.0) > 1e-6 for variable_id in variable_ids)
                has_alpha = any(variable_id.startswith(ALPHA_VAR_PREFIX) and primary_result.get(variable_id.removeprefix(ALPHA_VAR_PREFIX), 0.0) > 1e-6 for variable_id in variable_ids)

                # Keep COBRA-k's derived kinetic-difference column disabled. Native
                # NLP results may omit reaction-specific variables, while that helper
                # requires a complete kappa/gamma set. The individual columns below
                # safely display blanks for variables that were not part of the NLP.
                self._cobrak_excel_datasets[column_label] = OptimizationDataset(
                    data=primary_result,
                    with_df=has_df,
                    with_vplus=has_enzyme,
                    with_kappa=has_kappa,
                    with_gamma=has_gamma,
                    with_iota=has_iota,
                    with_alpha=has_alpha,
                    with_efficiency_coefficient=has_enzyme and any((has_kappa, has_gamma, has_iota, has_alpha)) and bool(primary_result.get(PROT_POOL_REAC_NAME)),
                )
                try:
                    create_cobrak_spreadsheet(
                        path=cobrak_output_path,
                        cobrak_model=self.solver._prepare_cobrak_model(),
                        variability_datasets={},
                        optimization_datasets=self._cobrak_excel_datasets,
                        show_regulation_coefficients=True,
                    )
                    logger.info("[%s] Generated side-by-side COBRA-k spreadsheet at '%s'.", scenario_id, cobrak_output_path)
                except Exception as exc:
                    logger.error("[%s] Failed to generate the side-by-side COBRA-k spreadsheet: %s", scenario_id, exc, exc_info=True)
        return output_path

    def save_environment_component_yaml(
        self,
        environment_name: str,
        environment_payload: dict[str, Any],
        scenario_id: str,
        filename_suffix: str = "environment",
    ) -> None:
        """Persist one reusable environment component as a YAML file.

        Args:
            environment_name: Environment component name written under the
                top-level ``environments`` key.
            environment_payload: Serialized environment block.
            scenario_id: Scenario identifier used in the output filename.
            filename_suffix: Suffix used for the YAML filename.
        """
        filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, scenario_id, filename_suffix, extension="yml")
        try:
            with open(filepath, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {"environments": {environment_name: environment_payload}},
                    handle,
                    sort_keys=False,
                    allow_unicode=False,
                    default_flow_style=None,
                )
            logger.info(f"[{self.experiment_def.experiment_name}] Saved environment component YAML to {filepath}")
        except Exception as e:
            logger.error(f"[{self.experiment_def.experiment_name}] Failed to save environment component YAML: {e}")

    def build_scenario_payload(
        self,
        scenario: structs.BiologicalScenario,
        *,
        status: str = "Running",
    ) -> dict[str, Any]:
        """Create the standard scenario-level payload shared by experiments.

        Args:
            scenario: Biological scenario currently being processed.
            status: Initial scenario status.

        Returns:
            dict[str, Any]: Standard scenario payload skeleton.
        """
        return {
            "experiment_name": self.experiment_def.experiment_name,
            "scenario_id": scenario.scenario_id,
            "substrate": scenario.substrate_name,
            "product": scenario.product_name,
            "diagnostics_environment": scenario.diagnostics_environment,
            "metrics": {},
            "steps": [],
            "status": status,
        }

    def resolve_scenario_metadata(
        self,
        scenario: structs.BiologicalScenario,
    ) -> dict[str, Any]:
        """Resolve common scenario identifiers shared by experiment classes.

        Args:
            scenario: Biological scenario currently being processed.

        Returns:
            dict[str, Any]: Common metadata used by experiment workflows.
        """
        product_name = scenario.product_name
        product_id = None
        if product_name and product_name in scenario.environment_conf.products:
            product_id = scenario.environment_conf.products[product_name].get("id")

        substrate_name = scenario.substrate_name
        substrate_ids = [
            item.get("id")
            for item in scenario.environment_conf.substrates.get(substrate_name, [])
            if item.get("id")
        ]

        return {
            "biomass_id": scenario.model_conf.biomass_id,
            "product_name": product_name,
            "product_id": product_id,
            "substrate_name": substrate_name,
            "substrate_ids": substrate_ids,
        }

    def compute_max_theoretical_yield(
        self,
        scenario: structs.BiologicalScenario,
        product_id: str | None,
    ) -> float:
        """Compute the maximum theoretical product yield when a product exists."""
        if not product_id:
            return 0.0
        return get_max_yield(scenario.model, product_reaction_id=product_id)

    def save_to_csv(self) -> None:
        """
        Generates a separate CSV file for every scenario combination processed.
        Each row in the CSV represents an individual optimization step,
        matching the desired standard format.
        """
        if not self.aggregate_results:
            logger.warning(f"[{self.experiment_def.experiment_name}] No results to save to CSV.")
            return

        exp_name = self.experiment_def.experiment_name
        scenario_ids = [str(payload.get("scenario_id", "Unknown_Scenario")) for payload in self.aggregate_results]
        scenario_label_map = build_scenario_filename_label_map(exp_name, scenario_ids)

        for run_payload in self.aggregate_results:
            scenario_id = run_payload.get("scenario_id", "Unknown_Scenario")
            scenario_label = self._scenario_filename_label(str(scenario_id), scenario_label_map)
            filepath = self._build_artifact_filepath(
                exp_name,
                scenario_label,
                extension="csv",
                source_id=f"csv:{scenario_id}",
                deduplicate=False,
            )
            rows = self._build_csv_rows(run_payload)
            df = self._build_csv_dataframe(rows)
            
            try:
                # Use semicolon separator matching your requested files
                df.to_csv(filepath, sep=";", index=False)
                logger.info(f"[{scenario_id}] Saved step-by-step CSV to {filepath}")
            except Exception as e:
                logger.error(f"[{scenario_id}] Failed to save CSV: {e}")

    def _build_csv_rows(self, run_payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate one scenario payload into the row dictionaries used for CSV output."""
        substrate = run_payload.get("substrate", "Unknown_Substrate")
        product = run_payload.get("product", "Unknown_Product")
        exp_name = self.experiment_def.experiment_name
        steps = run_payload.get("steps", [])

        if not steps:
            steps = [{"Step_Name": "Initialization_Error", "Status": run_payload.get("status", "Failed")}]

        rows: list[dict[str, Any]] = []
        for step in steps:
            row = {
                "substrate_condition": substrate,
                "target_product": product,
                "configuration_name": exp_name,
                "Step_Name": step.get("Step_Name", "N/A"),
                "Objective": step.get("Objective", "N/A"),
                "Objective_Direction": step.get("Objective_Direction", "N/A"),
                "Constraints": step.get("Constraints", "N/A"),
                "Status": step.get("Status", "N/A"),
                "Objective_Value": step.get("Objective_Value", "N/A"),
            }
            for k, v in step.items():
                if k not in row and k not in {"Post_Analyses", "Substrate_Uptake_Realized"}:
                    row[k] = v
            rows.append(row)
        return rows

    def _build_csv_dataframe(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Create the ordered CSV dataframe used by normal and fragment exports."""
        df = pd.DataFrame(rows)
        standard_cols = [
            "substrate_condition", "target_product", "configuration_name",
            "Step_Name", "Objective", "Objective_Direction", "Constraints",
            "Status", "Objective_Value"
        ]
        existing_standard = [c for c in standard_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in standard_cols]
        return df[existing_standard + other_cols]


    def generate_cnapy_files(
        self,
        model: cobra.Model,
        raw_solution: structs.SolverResult | dict[str, Any] | None,
        scenario_id: str,
        objective_direction: str = "max",
        filename_suffix: str = "cnapy",
        flux_bounds: dict[str, float | tuple[float, float] | list[float]] | None = None,
    ) -> None:
        """
        Generates CNApy scenario files for manual exploration in the CNApy GUI.

        COBRA-k results additionally receive a native comparison file with the
        ``_cobrak`` suffix. The original exact and tolerant files are retained.

        Args:
            model: The base cobrapy Model object.
            raw_solution: A `SolverResult` or dumped solution dictionary.
            scenario_id: The specific scenario string to ensure unique filenames.
            objective_direction: Optimization direction stored in the scenario metadata.
            filename_suffix: Suffix used for the exported `.scen` filename.
            flux_bounds: Optional precomputed reaction bounds to export directly
                instead of deriving fluxes from a solver solution.
        """
        filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, scenario_id, filename_suffix, extension="scen")
        tolerant_filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, scenario_id, f"{filename_suffix}_tolerant", extension="scen")
        cobrak_filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, scenario_id, f"{filename_suffix}_cobrak", extension="scen")
        logger.info(f"[{scenario_id}] Generating CNApy files: {filepath}")
        
        try:
            if flux_bounds is not None:
                # PATH A: Export caller-provided bounds without consulting a
                # solver result. This path is backend-neutral.
                logger.debug("[%s] Building CNApy scenario from %d explicit flux bound(s).", scenario_id, len(flux_bounds))
                structure = {
                    "fluxes": {
                        reaction_id: (
                            [float(bounds), float(bounds)]
                            if isinstance(bounds, (int, float))
                            else [float(bounds[0]), float(bounds[1])]
                        )
                        for reaction_id, bounds in flux_bounds.items()
                    },
                    "pinned_reactions": [],
                    "description": "",
                    "objective_direction": objective_direction,
                    "objective_coefficients": {},
                    "use_scenario_objective": False,
                    "reactions": {},
                    "constraints": [],
                    "version": 2,
                }
            else:
                # PATH B: Use the existing backend-neutral exporter first so
                # its output remains available for direct comparison.
                logger.debug("[%s] Building CNApy scenario through the backend-neutral exporter.", scenario_id)
                flux_data = preprocess_solution_for_cnapy(raw_solution, model)
                if not flux_data:
                    logger.warning(f"[{scenario_id}] No valid flux data extracted. Skipping CNApy file.")
                    return
                structure = generate_cnapy_scenario(flux_data, opt_direction=objective_direction)
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(structure, handle, indent=2)

            tolerant_structure = self._build_tolerant_cnapy_structure(structure)
            with open(tolerant_filepath, "w", encoding="utf-8") as handle:
                json.dump(tolerant_structure, handle, indent=2)
            logger.info(f"[{scenario_id}] Generated tolerant CNApy file: {tolerant_filepath}")

            # PATH C: Add COBRA-k's native export without replacing either of
            # the existing MMP2 CNApy files.
            if isinstance(raw_solution, structs.SolverResult) and raw_solution.native_result and hasattr(getattr(self, "solver", None), "export_cnapy_scenario"):
                self.solver.export_cnapy_scenario(cobrak_filepath, raw_solution)
                logger.info("[%s] Generated side-by-side COBRA-k CNApy file at '%s'.", scenario_id, cobrak_filepath)
        except Exception as e:
            logger.error(f"[{scenario_id}] Failed to generate CNApy files: {e}")

    @staticmethod
    def _build_tolerant_cnapy_structure(
        structure: dict[str, Any],
        epsilon: float = CNAPY_TOLERANT_EXPORT_EPSILON,
        zero_threshold: float = CNAPY_TOLERANT_ZERO_THRESHOLD,
    ) -> dict[str, Any]:
        """Build a CNApy scenario tolerant to small numerical differences.

        The input scenario is deep-copied so callers can safely retain or write
        the exact solver export. Reaction bounds close to zero are centered on
        zero before widening; all other bounds are expanded outward by the
        configured epsilon. Non-flux CNApy metadata is preserved unchanged.

        Args:
            structure: CNApy scenario dictionary containing a ``fluxes``
                mapping and optional metadata such as description, objective,
                pinned reactions, and additional constraints.
            epsilon: Absolute amount subtracted from each lower bound and added
                to each upper bound.
            zero_threshold: Absolute threshold below which both bounds are
                treated as numerical zero and replaced by
                ``[-epsilon, epsilon]``.

        Returns:
            Deep-copied CNApy scenario with widened flux bounds and a
            description documenting the applied tolerance.

        Raises:
            TypeError: If a flux bound cannot be converted to ``float``.
            ValueError: If a flux bound contains a non-numeric string or other
                value that ``float`` cannot parse.
            IndexError: If a flux entry does not contain lower and upper bounds.

        Notes:
            This method does not validate reaction IDs against a model. Model
            compatibility is handled by the caller or native solver exporter.
        """
        # STEP 1: Preserve the exact input scenario and all non-flux metadata.
        tolerant_structure = copy.deepcopy(structure)

        # STEP 2: Record the tolerance transformation in the scenario text so
        # exact and tolerant artifacts remain distinguishable in CNApy.
        tolerant_structure["description"] = (
            f"{structure.get('description', '')}\n"
            f"Tolerant export: flux bounds widened by +/- {epsilon:g}; "
            f"bounds within +/- {zero_threshold:g} are centered on zero before widening."
        ).strip()

        # STEP 3: Widen every reaction interval independently. Near-zero
        # intervals are centered first to avoid preserving solver-sign noise.
        tolerant_fluxes: dict[str, list[float]] = {}
        for reaction_id, bounds in structure.get("fluxes", {}).items():
            lower = float(bounds[0])
            upper = float(bounds[1])

            if abs(lower) <= zero_threshold and abs(upper) <= zero_threshold:
                tolerant_fluxes[reaction_id] = [-epsilon, epsilon]
            else:
                tolerant_fluxes[reaction_id] = [lower - epsilon, upper + epsilon]

        # STEP 4: Replace only the copied flux map and return the completed
        # tolerant scenario structure.
        tolerant_structure["fluxes"] = tolerant_fluxes
        return tolerant_structure


    def save_raw_solutions(self, solution: structs.SolverResult | dict[str, float], scenario_id: str, suffix: str = "") -> None:
        """Save normalized values and any native backend stages.

        Failed solves often have no normalized values but still contain the
        native solver result needed to diagnose their termination. That native
        payload is therefore persisted independently.

        Args:
            solution: Shared solver result or a raw value dictionary.
            scenario_id: Identifier used in the artifact filename.
            suffix: Optional filename suffix.
        """
        solution_values = solution.values if isinstance(solution, structs.SolverResult) else solution
        native_result = solution.native_result if isinstance(solution, structs.SolverResult) else None
        if not solution_values and not native_result:
            logger.warning(f"[{scenario_id}] No raw solutions to save.")
            return

        safe_suffix = f"_{suffix}" if suffix else ""
        filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, f"{scenario_id}{safe_suffix}", "raw_solution", extension="json")
        
        try:
            if solution_values:
                with open(filepath, "w", encoding="utf-8") as handle:
                    json.dump(solution_values, handle, indent=4)
                logger.debug("[%s] Saved raw solver solutions to %s", scenario_id, filepath)
            if native_result:
                # Native COBRA-k dictionaries can be much larger than the
                # normalized result. Store them separately and compressed so
                # ordinary JSON/CSV artifacts remain backend-neutral.
                native_filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, f"{scenario_id}{safe_suffix}", "cobrak_native", extension="json.gz")
                logger.info("[%s] Saving compressed native solver stages: stages=%s, path='%s'.", scenario_id, list(native_result), native_filepath)
                with gzip.open(native_filepath, "wt", encoding="utf-8") as handle:
                    json.dump(native_result, handle, default=str)
                logger.debug("[%s] Saved compressed COBRA-k native result to %s", scenario_id, native_filepath)
        except Exception as e:
            logger.error(f"[{scenario_id}] Failed to save raw solutions: {e}")

    def save_realized_pathway_adjustments(
        self,
        solution_values: dict[str, float],
        scenario_id: str,
        *,
        forward_reaction_suffix: str,
        reverse_reaction_suffix: str,
    ) -> None:
        """
        Persist realized pathway selections or implied adjustment knockouts.

        If exclusive reaction groups are available, this exporter writes a
        reusable ``*_realized_adjustments.json`` file that blocks the
        unselected alternatives with ``[0, 0]`` bounds. Otherwise it falls back
        to saving the active ``z_var_*`` selections in a
        ``*_realized_pathways.json`` file for debugging.

        Args:
            solution_values: Raw solver values keyed by variable id.
            scenario_id: Scenario identifier used in the output filename.
            forward_reaction_suffix: Forward-direction reaction suffix used by
                the current model.
            reverse_reaction_suffix: Reverse-direction reaction suffix used by
                the current model.
        """
        if not solution_values:
            return

        active_z_vars = {var_name: float(val) for var_name, val in solution_values.items() if var_name.startswith("z_var_") and val is not None and val > 0.5}
        all_z_var_reaction_ids = [var_name.replace("z_var_", "", 1) for var_name in solution_values if var_name.startswith("z_var_")]
        if not all_z_var_reaction_ids:
            logger.debug(f"[{scenario_id}] No pathway-selection z_vars found.")
            return

        def base_reaction_id(reaction_id: str) -> str:
            if forward_reaction_suffix and reaction_id.endswith(forward_reaction_suffix):
                return reaction_id[: -len(forward_reaction_suffix)]
            if reverse_reaction_suffix and reaction_id.endswith(reverse_reaction_suffix):
                return reaction_id[: -len(reverse_reaction_suffix)]
            return reaction_id

        exclusive_reaction_groups = []
        if getattr(self.experiment_def, "biological_scenarios", None):
            exclusive_reaction_groups = list(self.experiment_def.biological_scenarios[0].model_conf.exclusive_reaction_groups or [])

        realized_adjustments: dict[str, list[float]] = {}
        if exclusive_reaction_groups:
            active_base_ids = {base_reaction_id(var_name.replace("z_var_", "", 1)) for var_name in active_z_vars}
            for group in exclusive_reaction_groups:
                group_prefixes = [str(prefix) for prefix in group]
                group_reaction_ids = [reaction_id for reaction_id in all_z_var_reaction_ids if any(base_reaction_id(reaction_id) == prefix or base_reaction_id(reaction_id).startswith(f"{prefix}_") for prefix in group_prefixes)]
                if not group_reaction_ids:
                    continue

                chosen_base_ids = {base_reaction_id(reaction_id) for reaction_id in group_reaction_ids if base_reaction_id(reaction_id) in active_base_ids}
                for reaction_id in group_reaction_ids:
                    reaction_base_id = base_reaction_id(reaction_id)
                    if chosen_base_ids and reaction_base_id not in chosen_base_ids:
                        realized_adjustments[reaction_id] = [0.0, 0.0]
                    elif not chosen_base_ids and "_VARIANT_" in reaction_base_id:
                        realized_adjustments[reaction_id] = [0.0, 0.0]

        artifact_suffix = "realized_adjustments" if realized_adjustments else "realized_pathways"
        filepath = self._build_artifact_filepath(self.experiment_def.experiment_name, scenario_id, artifact_suffix, extension="json")
        payload = realized_adjustments if realized_adjustments else active_z_vars
        if not payload:
            logger.debug(f"[{scenario_id}] No realized pathway artifact payload could be derived.")
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            logger.info(f"[{scenario_id}] Saved realized pathway artifact to {filepath}")
        except Exception as e:
            logger.error(f"[{scenario_id}] Failed to save realized pathway artifact: {e}")


    def build_standard_step_payload(
        self,
        step_name: str,
        objective_var_ids: list[str],
        objective_direction: str,
        constraints: list[list[Any]],
        solution: Any,  # SolverResult
        biomass_id: str,
        product_id: str | None,
        swaps_required: Any = "N/A",
        substrate_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the shared tabular payload schema for one workflow step.

        Args:
            step_name: Human-readable workflow step identifier.
            objective_var_ids: Objective variable ids optimized in this step.
            objective_direction: Objective sense such as ``"max"`` or
                ``"min"``.
            constraints: Solver-ready constraint block active for the step.
            solution: Solver result or solver-like object with ``status``,
                ``objective_value``, and ``values`` attributes.
            biomass_id: Scenario-specific biomass reaction id.
            product_id: Scenario-specific product reaction id, if available.
            swaps_required: Optional realized swap count for swap-aware
                experiments.
            substrate_ids: Environment reactions representing the active
                substrate condition.

        Returns:
            dict[str, Any]: Standardized payload row used by JSON and CSV
            reporting across experiment classes.

        Examples:
            ``build_standard_step_payload("optATPM", ["ATPM"], "max", ... )``
            produces a row with the common columns ``Step_Name``,
            ``Objective``, ``Status``, ``Biomass``, ``Product_Flux``, and
            ``ATPM``.
        """
        is_opt = (solution.status or "").lower() == "optimal"

        # Format constraints for readability
        formatted_constraints = "; ".join([str(c) for c in constraints]) if constraints else ""

        metadata = dict(getattr(solution, "metadata", {}) or {})
        metrics = dict(getattr(solution, "metrics", {}) or {})
        uptake_substrate_realized = {}
        if is_opt:
            for substrate_id in substrate_ids or []:
                flux = solution.values.get(substrate_id)
                try:
                    flux = float(flux)
                    uptake_substrate_realized[substrate_id] = {
                        "flux": flux,
                        "uptake_rate": abs(flux),
                    }
                except (TypeError, ValueError):
                    uptake_substrate_realized[substrate_id] = {"flux": None, "uptake_rate": None}
                    logger.warning("Optimal solution does not contain a numeric flux for configured substrate reaction '%s'.", substrate_id)
            logger.info("Realized substrate uptake for step '%s': %s", step_name, uptake_substrate_realized)
        return {
            "Step_Name": step_name,
            "Objective": ", ".join(objective_var_ids),
            "Objective_Direction": objective_direction,
            "Constraints": formatted_constraints,
            "Status": solution.status,
            "Objective_Value": solution.objective_value if is_opt else "N/A",
            
            # Standard Universal Columns
            "MinForce": solution.values.get("var_B", "N/A") if is_opt else "N/A",
            "Biomass": solution.values.get(biomass_id, "N/A") if is_opt else "N/A",
            "Product_Flux": solution.values.get(product_id, "N/A") if product_id and is_opt else "N/A",
            "Substrate_Uptake_Realized": uptake_substrate_realized,
            "ATPM": solution.values.get("ATPM", "N/A") if is_opt else "N/A",
            "Swaps_Required": swaps_required,

            # Solver diagnostics are retained for both successful and failed
            # runs so campaign summaries expose where an NLP stopped.
            "Solver_Status": metadata.get("solver_status"),
            "Solver_Status_Name": metadata.get("solver_status_name"),
            "Termination_Condition": metadata.get("termination_condition"),
            "Termination_Condition_Name": metadata.get("termination_condition_name"),
            "Solver_Message": getattr(solution, "message", "") or None,
            "Solver_Error": getattr(solution, "error_message", "") or None,
            "LP_Seed_Objective": metadata.get("lp_seed_objective"),
            "Retained_Objective_Fraction": metadata.get("retained_objective_fraction"),
            "Retained_Objective_Lower": metadata.get("retained_objective_lower"),
            "Retained_Objective_Upper": metadata.get("retained_objective_upper"),
            "Variability_Variables": metadata.get("variability_variables"),
            "Removed_Inactive_Reactions": metadata.get("removed_inactive_reactions"),
            "Active_Reactions_Before": metrics.get("active_reactions_before"),
            "Active_Reactions_After": metrics.get("active_reactions_after"),
            "Floored_Reaction_Ranges": metrics.get("variability_floored_reaction_ranges"),
            "Active_Set_Source": metadata.get("active_set_source"),
            "FVA_Directions_Resolved_From_LP_Seed": metadata.get("fva_directions_resolved_from_lp_seed"),
            "LP_Seed_Active_Reactions": metrics.get("lp_seed_active_reactions"),
            "LP_Resolved_Thermodynamic_Pairs": metrics.get("lp_resolved_thermodynamic_pairs"),
            "LP_Unselected_Thermodynamic_Pairs": metrics.get("lp_unselected_thermodynamic_pairs"),
            "LP_Unselected_Direction_Override_Count": metrics.get("lp_unselected_direction_override_count"),
            "LP_Unselected_Direction_Overrides": metadata.get("lp_unselected_direction_overrides"),
            "Thermodynamic_Bidirectional_Pairs": metrics.get("thermodynamic_bidirectional_pairs"),
            "Active_Set_Union_Seed_File": metadata.get("active_set_union_seed_file"),
            "Active_Set_Union_Reactions": metrics.get("active_set_union_reactions"),
            "Active_Set_Union_Added_Reactions": metrics.get("active_set_union_added_reactions"),
            "Initialized_From_LP_Seed": metadata.get("initialized_from_lp_seed"),
            "Initialized_NLP_Variables": metrics.get("initialized_nlp_variables"),
            "Clamped_NLP_Initial_Values": metrics.get("clamped_nlp_initial_values"),
            "NLP_Multistart_Attempts": metrics.get("nlp_multistart_attempts"),
            "NLP_Multistart_Successes": metrics.get("nlp_multistart_successes"),
            "NLP_Multistart_Selected_Attempt": metrics.get("nlp_multistart_selected_attempt"),
            "NLP_Multistart_Seed": metadata.get("nlp_multistart_seed"),
            "NLP_Multistart_Objectives": metadata.get("nlp_multistart_objectives"),
            "Solution_Load_Error": metadata.get("solution_load_error"),
        }

    @staticmethod
    def attach_post_analysis_results(
        step_payload: dict[str, Any],
        post_analysis_results: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Attach dispatcher results in two forms:
        - nested under `Post_Analyses` for the JSON payload
        - flattened into the step row so CSV/reporting code can read them directly
        """
        results = dict(post_analysis_results or {})
        step_payload["Post_Analyses"] = results

        for key, value in results.items():
            if key not in step_payload:
                step_payload[key] = value

        return step_payload

    @staticmethod
    def add_key_reaction_placeholders(
        step_payload: dict[str, Any],
        key_reaction_prefixes: list[str] | None,
    ) -> dict[str, Any]:
        """Initialize standard key-reaction columns with placeholder values.

        Args:
            step_payload: Step payload that will later be enriched in place.
            key_reaction_prefixes: Configured reaction prefixes of interest.

        Returns:
            dict[str, Any]: The same payload updated with placeholder columns.
        """
        for prefix in key_reaction_prefixes or []:
            step_payload[f"variant_{prefix}"] = "N/A"
            step_payload[f"flux_{prefix}"] = "N/A"

        return step_payload


    def _get_key_fluxes(self, solution, key_reaction_prefixes: list[str]) -> dict[str, tuple[str, float]]:
        """
        Intelligently finds the specific active variant of a reaction given a prefix,
        and calculates its net flux (FWD - REV).
        
        Returns a dictionary mapping the prefix to a tuple of (active_variant_name, net_flux).
        """
        results = {}
        is_optimal = (solution.status or "").lower() == "optimal"
        
        if not is_optimal:
            return {prefix: ('N/A', "N/A") for prefix in key_reaction_prefixes}
            
        # Raw values to check z_vars
        raw_values = solution.values
        
        for prefix in key_reaction_prefixes:
            active_variant = f"{prefix}"  # Fallback to just the prefix
            found_active = False
            
            # 1. Search for an active z_var matching the prefix
            for var_name, var_value in raw_values.items():
                if var_name.startswith(f"z_var_{prefix}") and var_value > 0.5:
                    # Strip 'z_var_' and directional suffixes to get the base variant ID
                    full_id = var_name.replace("z_var_", "")
                    active_variant = full_id.replace("_FWD", "").replace("_REV", "")
                    found_active = True
                    break
            
            if not found_active:
                logger.debug(f"No active z_var found for prefix '{prefix}'. Assuming default/knocked-out.")
                
            # 2. Calculate Net Flux (FWD - REV)
            fwd_flux = raw_values.get(f"{active_variant}_FWD", raw_values.get(active_variant, 0.0))
            rev_flux = raw_values.get(f"{active_variant}_REV", 0.0)
            net_flux = fwd_flux - rev_flux
            
            # Only round if it's a number to avoid tiny floating point errors (e.g., 1e-13)
            if isinstance(net_flux, float) and abs(net_flux) < 1e-9:
                net_flux = 0.0
                
            results[prefix] = (active_variant, net_flux)

        return results

    @staticmethod
    def add_key_metabolite_placeholders(
        step_payload: dict[str, Any],
        key_metabolite_ids: list[str] | None,
    ) -> dict[str, Any]:
        """Initialize standard key-metabolite columns with placeholder values.

        Args:
            step_payload: Step payload that will later be enriched in place.
            key_metabolite_ids: Configured metabolite IDs of interest.

        Returns:
            dict[str, Any]: The same payload updated with placeholder columns.
        """
        for met_id in key_metabolite_ids or []:
            step_payload[f"conc_{met_id}"] = "N/A"

        return step_payload

    def _get_key_concentrations(self, solution, key_metabolite_ids: list[str]) -> dict[str, float | str]:
        """
        Reads the realized concentration of each configured metabolite from the
        current solution's log-concentration variable (x_<metabolite_id>), converting
        back out of log-space with exp().

        Returns a dictionary mapping the metabolite ID to its concentration, or "N/A"
        if the solve wasn't optimal or the metabolite has no concentration variable
        (e.g. the step wasn't thermodynamically formulated).
        """
        results: dict[str, float | str] = {}
        is_optimal = (solution.status or "").lower() == "optimal"

        if not is_optimal:
            return {met_id: "N/A" for met_id in key_metabolite_ids}

        raw_values = solution.values

        for met_id in key_metabolite_ids:
            log_conc = raw_values.get(f"x_{met_id}")
            results[met_id] = exp(log_conc) if log_conc is not None else "N/A"

        return results
