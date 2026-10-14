from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import mymodelingpackage.structures as structs
from mymodelingpackage.analysis_config_setup import build_analysis_config
from mymodelingpackage.experiments.characterization import CharacterizationExperiment
from mymodelingpackage.experiments.fluxvalidation import FluxValidationExperiment
from mymodelingpackage.experiments.variablescan import VariableScanExperiment
from mymodelingpackage.scenario_setup import ScenariosBuilder
from mymodelingpackage.solvers.stoichiometric import StoichiometricSolver
from mymodelingpackage.solvers.thermo import ThermoSolverCPLEX
from mymodelingpackage.utilities import assemble_settings, generate_summary_csv

logger = logging.getLogger(__name__)


class Campaign:
    """Orchestrates typed campaign definitions.

    Attributes:
        campaign_definition: Typed campaign specification to execute.
        generate_summary: Whether to generate a campaign summary CSV after the
            run completes.
        generate_summary_excel: Whether to generate a campaign Summary/Fluxes
            workbook after all experiments finish.
        continue_on_error: Whether to continue running later experiments after
            one experiment fails.
        campaign_output_dir: Root output directory for campaign artifacts.
    """

    _EXPERIMENT_CLASS_MAP = {"characterization": CharacterizationExperiment, "fluxvalidation": FluxValidationExperiment, "variablescan": VariableScanExperiment}
    _ALLOWED_SOLVER_KINDS = {"characterization": {"thermo", "stoichiometric", "cobrak"}, "fluxvalidation": {"stoichiometric"}, "variablescan": {"thermo"}}

    def __init__(
        self,
        campaign_definition: structs.CampaignDefinition,
        overwrite_output: bool = False,
        generate_summary: bool = False,
        generate_summary_excel: bool = False,
        continue_on_error: bool = False,
    ) -> None:
        """Initializes a campaign runner from a typed campaign definition.

        Args:
            campaign_definition: Fully typed campaign configuration to execute.
            overwrite_output: Whether to remove an existing campaign output
                directory before creating a new one.
            generate_summary: Whether to generate a summary CSV after the full
                campaign finishes.
            generate_summary_excel: Whether to generate a campaign workbook
                containing combined Summary and available Fluxes sheets.
            continue_on_error: Whether to continue running remaining
                experiments after a failure.
        """
        self.campaign_definition = campaign_definition
        self.generate_summary = generate_summary
        self.generate_summary_excel = generate_summary_excel
        self.continue_on_error = continue_on_error
        self.campaign_output_dir = Path(self.campaign_definition.output_dir) / self.campaign_definition.campaign_name

        logger.debug("Initializing campaign '%s' at '%s' (overwrite_output=%s, generate_summary=%s, generate_summary_excel=%s, continue_on_error=%s).", self.campaign_definition.campaign_name, self.campaign_output_dir, overwrite_output, self.generate_summary, self.generate_summary_excel, self.continue_on_error)

        # Campaign construction owns the top-level output directory so every
        # later execution path can assume the campaign root already exists.
        if overwrite_output and self.campaign_output_dir.exists():
            logger.info("Removing existing campaign output directory '%s' before rerun.", self.campaign_output_dir)
            shutil.rmtree(self.campaign_output_dir)
        self.campaign_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Campaign output directory ready at '%s'.", self.campaign_output_dir)

    @classmethod
    def _validate_solver_compatibility(
        cls,
        *,
        experiment_type: str,
        solver_kind: str,
        experiment_name: str,
    ) -> None:
        """Fail fast when an experiment is configured with an incompatible solver."""
        normalized_experiment_type = str(experiment_type or "").strip().lower()
        normalized_solver_kind = str(solver_kind or "").strip().lower()
        allowed_solver_kinds = cls._ALLOWED_SOLVER_KINDS.get(normalized_experiment_type)
        if allowed_solver_kinds is None:
            return
        if normalized_solver_kind in allowed_solver_kinds:
            return

        allowed_list = ", ".join(sorted(allowed_solver_kinds))
        raise ValueError(
            f"Experiment '{experiment_name}' of type '{normalized_experiment_type}' is incompatible with solver "
            f"'{normalized_solver_kind}'. Allowed solver kind(s): {allowed_list}."
        )

    # ---------------------------------------------------------------------
    # Campaign builder (legacy path)
    # ---------------------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        campaign_to_run: str,
        all_components: dict,
        output_root: str = "./Results",
        overwrite_output: bool = False,
        generate_summary: bool = False,
        generate_summary_excel: bool = False,
        continue_on_error: bool = False,
        models_dir: str | None = None,
        data_dir: str | None = None,
        selected_experiment_index: int | None = None,
        selected_scenario_index: int | None = None,
    ) -> "Campaign":
        """Builds a campaign from modular component dictionaries.

        Args:
            campaign_to_run: Name of the campaign entry to resolve from the
                ``Campaigns`` block.
            all_components: Raw component dictionary produced from the modular
                YAML structure.
            output_root: Root directory under which campaign results are stored.
            overwrite_output: Whether to remove an existing campaign output
                directory before creating a new one.
            generate_summary: Whether to generate a summary CSV after the full
                campaign finishes.
            generate_summary_excel: Whether to generate a campaign workbook
                containing combined Summary and available Fluxes sheets.
            continue_on_error: Whether to continue running remaining
                experiments after a failure.
            models_dir: Optional models asset directory forwarded to
                ``ScenariosBuilder``.
            data_dir: Optional data asset directory forwarded to
                ``ScenariosBuilder``.

        Returns:
            Campaign: A fully initialized campaign instance ready to run.

        Raises:
            ValueError: If the requested campaign is missing or a campaign item
                is structurally invalid.
            NotImplementedError: If an unsupported solver type is requested by
                the merged analysis settings.
        """
        logger.info("Building campaign '%s' from components with output_root='%s'.", campaign_to_run, output_root)
        experiments_list = []

        # Step 1: resolve the named campaign to the ordered list of triplets it
        # should execute. If the name is missing, the configuration is invalid
        # and there is nothing meaningful to run.
        campaigns_dict = all_components.get("Campaigns", {})
        if campaign_to_run not in campaigns_dict: 
            raise ValueError(f"Campaign '{campaign_to_run}' not found in the 'Campaigns' block of the YAML.")

        campaign_runs = campaigns_dict[campaign_to_run]
        logger.debug("Campaign '%s' contains %d experiment triplets.", campaign_to_run, len(campaign_runs))

        # Step 2: turn each [model, environment, analysis] triplet into one
        # fully typed ExperimentDefinition that the runner can execute later.
        for idx, run_item in enumerate(campaign_runs):
            if selected_experiment_index is not None and idx != selected_experiment_index:
                continue
            components = run_item.get("components", [])
            if len(components) != 3: 
                raise ValueError(f"Campaign '{campaign_to_run}' item {idx} is invalid. Expected 3 components [model, environment, analysis], got: {components}")

            model_key, environment_key, analysis_key = components
            logger.debug("Parsing campaign '%s' item %d as model='%s', environment='%s', analysis='%s'.", campaign_to_run, idx, model_key, environment_key, analysis_key)

            # We reuse the legacy merge path so x-* indirections are resolved
            # before anything is validated into typed model/environment configs.
            merged_settings = assemble_settings(components, all_components)
            experiment_name = f"{model_key}_{environment_key}_{analysis_key}"
            logger.debug("Merged %d settings keys for experiment '%s'.", len(merged_settings), experiment_name)

            try:
                model_config_obj = structs.ModelConfig(**merged_settings)
                environment_config_obj = structs.EnvironmentConfig(**merged_settings)
            except Exception as e:
                logger.error("Failed validating model/environment config for experiment '%s': %s", experiment_name, e)
                raise

            # Step 3: detect which experiment type block exists after merging.
            # We support both nested blocks and a flatter legacy `type` form.
            exp_type_key = None
            flat_analysis_type = ""
            for key in merged_settings.keys():
                if key.lower() in cls._EXPERIMENT_CLASS_MAP:
                    exp_type_key = key
                    break

            analysis_type_name = ""
            if not exp_type_key:
                flat_analysis_type = str(merged_settings.get("type", "")).strip().lower()
                if flat_analysis_type in cls._EXPERIMENT_CLASS_MAP:
                    exp_type_key = flat_analysis_type
                    analysis_type_name = str(merged_settings.get("type", "")).strip()
            elif "type" in merged_settings:
                analysis_type_name = str(merged_settings.get("type", "")).strip()

            if not exp_type_key:
                valid_types = list(cls._EXPERIMENT_CLASS_MAP.keys())
                raise ValueError(f"Analysis '{analysis_key}' is missing a recognized experiment type block. Expected one of {valid_types} (case-insensitive) as a key, but found: {list(merged_settings.keys())}")

            # Solver settings live inside the experiment block after the merge.
            # For flat legacy configs we reconstruct the analysis-only portion by
            # dropping model/environment fields from the merged settings.
            if flat_analysis_type:
                model_fields = set(structs.ModelConfig.model_fields.keys())
                environment_fields = set(structs.EnvironmentConfig.model_fields.keys())
                inner_analysis_conf = {key: value for key, value in merged_settings.items() if key not in model_fields and key not in environment_fields and key != "type"}
            else:
                inner_analysis_conf = merged_settings.get(exp_type_key, {})

            if str(exp_type_key).strip().lower() == "mcs":
                default_solver_type = "mcs"
            elif str(exp_type_key).strip().lower() == "dynamicfba":
                default_solver_type = "stoichiometric"
            else:
                default_solver_type = "thermo"
            solver_type = str(inner_analysis_conf.get("solver", default_solver_type)).strip().lower()
            normalized_solver_type = solver_type.replace("-", "_").replace(" ", "_")
            solver_kwargs = inner_analysis_conf.get("solver_settings", {})
            logger.debug("Experiment '%s' resolved analysis type '%s' and solver '%s' with solver setting keys=%s.", experiment_name, exp_type_key, solver_type, list(solver_kwargs.keys()))

            if normalized_solver_type == "thermo":
                solver_config = structs.ThermoSolverCPLEXConfig(**solver_kwargs)
            elif normalized_solver_type == "stoichiometric":
                solver_config = structs.StoichiometricSolverConfig(**solver_kwargs)
            elif normalized_solver_type in {"cobrak", "cobra_k", "cobra-k"}:
                solver_config = structs.CobrakSolverConfig(**solver_kwargs)
            elif normalized_solver_type in {"mcs", "optlang"}:
                solver_config = structs.MCSSolverOPTLANGConfig(**solver_kwargs)
            elif normalized_solver_type in {"mcs_sd", "strain_design", "straindesign"}:
                solver_config = structs.MCSSolverSDConfig(**solver_kwargs)
            else:
                raise NotImplementedError(f"Solver type '{solver_type}' parsing is not yet implemented.")

            cls._validate_solver_compatibility(
                experiment_type=exp_type_key,
                solver_kind=solver_config.to_solve,
                experiment_name=experiment_name,
            )

            # Step 4: build the biological scenarios once from the typed
            # model/environment configuration so the experiment definition later
            # contains concrete scenario objects instead of raw config fragments.
            try:
                builder = ScenariosBuilder(models_dir=models_dir, data_dir=data_dir)
                logger.debug("Building scenarios for experiment '%s' using models_dir='%s' and data_dir='%s'.", experiment_name, models_dir, data_dir)
                scenarios = builder.build_all_scenarios(
                    model_component=model_key,
                    environment_component=environment_key,
                    model_config=model_config_obj,
                    environment_config=environment_config_obj,
                    include_cobrak_model=solver_config.to_solve == "cobrak",
                    scenario_indices={selected_scenario_index} if selected_scenario_index is not None else None,
                )
                if selected_scenario_index is not None and not scenarios:
                    raise IndexError(f"Selected scenario index {selected_scenario_index} is out of bounds for experiment '{experiment_name}'.")
            except Exception as e:
                logger.error("Failed building scenarios for experiment '%s': %s", experiment_name, e)
                raise
            logger.debug("Built %d biological scenarios for experiment '%s'.", len(scenarios), experiment_name)

            # Step 5: rebuild the typed analysis config from the merged block so
            # the execution layer sees the fully materialized analysis settings,
            # including any x-additional-analysis-params expansions.
            analysis_validation_conf = dict(inner_analysis_conf)
            analysis_validation_conf.pop("solver", None)
            analysis_validation_conf.pop("solver_settings", None)
            merged_analysis_components = {"analyses": {analysis_key: {analysis_type_name or exp_type_key: analysis_validation_conf}}}
            analysis_config = build_analysis_config(merged_analysis_components)[analysis_key]
            logger.debug("Rebuilt typed analysis config '%s' for experiment '%s'.", analysis_config.type, experiment_name)

            exp_def = structs.ExperimentDefinition(
                experiment_name=experiment_name,
                experiment_type=exp_type_key,
                solver_config=solver_config,
                biological_scenarios=scenarios,
                analysis_config=analysis_config,
            )
            experiments_list.append(exp_def)
            logger.info("Prepared experiment '%s' (%s) with %d scenarios.", experiment_name, exp_type_key, len(scenarios))

        campaign_def = structs.CampaignDefinition(campaign_name=campaign_to_run, output_dir=output_root, experiments=experiments_list)
        logger.info("Campaign '%s' prepared with %d experiments.", campaign_to_run, len(experiments_list))

        return cls(
            campaign_definition=campaign_def,
            overwrite_output=overwrite_output,
            generate_summary=generate_summary,
            generate_summary_excel=generate_summary_excel,
            continue_on_error=continue_on_error,
        )


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run_all(self) -> Dict[str, list[dict[str, Any]]]:
        """Runs all experiments in campaign order.

        Returns:
            Dict[str, list[dict[str, Any]]]: Mapping from experiment name to
            the list of payload rows returned by that experiment.
        """
        # Campaign execution stays deliberately simple: snapshot the typed plan,
        # then run experiments in declaration order so logs, output folders, and
        # returned results all describe the same sequence.
        logger.info("Starting campaign '%s' with %d experiments; output_dir='%s'.", self.campaign_definition.campaign_name, len(self.campaign_definition.experiments), self.campaign_output_dir)
        self._save_campaign_snapshot()

        results: Dict[str, list[dict[str, Any]]] = {}

        # Results are keyed by experiment name instead of index so the caller
        # gets a readable campaign summary structure back immediately.
        for experiment_index, experiment_def in enumerate(self.campaign_definition.experiments):
            experiment_name = experiment_def.experiment_name
            logger.info("Running experiment %d/%d: name='%s', type='%s', scenarios=%d.", experiment_index + 1, len(self.campaign_definition.experiments), experiment_name, experiment_def.experiment_type, len(experiment_def.biological_scenarios))
            results[experiment_name] = self._run_single_experiment(experiment_def=experiment_def, experiment_index=experiment_index)
            logger.debug("Stored %d payload rows for experiment '%s'.", len(results[experiment_name]), experiment_name)

        if self.generate_summary or self.generate_summary_excel:
            logger.info("Generating campaign summary in '%s'.", self.campaign_output_dir)
            generate_summary_csv(str(self.campaign_output_dir))

        if self.generate_summary_excel:
            # Combine the standard CSV summary with Fluxes sheets emitted by
            # BaseExperiment. Workbooks without steady-state fluxes still
            # contribute their normal summary rows through the campaign CSV.
            summary_path = self.campaign_output_dir / f"{self.campaign_output_dir.name}_summary.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path, sep=";")
            else:
                summary_rows = []
                for experiment_name, experiment_results in results.items():
                    for run_payload in experiment_results:
                        steps = run_payload.get("steps") or [{"Step_Name": "No_Results", "Status": run_payload.get("status", "Unknown")}]
                        for step in steps:
                            summary_row = {
                                "Task": experiment_name,
                                "scenario_id": run_payload.get("scenario_id", "Unknown_Scenario"),
                                "substrate_condition": run_payload.get("substrate", "Unknown_Substrate"),
                                "target_product": run_payload.get("product", "Unknown_Product"),
                                "configuration_name": experiment_name,
                            }
                            summary_row.update({key: value for key, value in step.items() if key != "Post_Analyses"})
                            summary_rows.append(summary_row)
                summary = pd.DataFrame(summary_rows)
            flux_frames = []
            experiment_workbook_names = {
                f"{experiment.experiment_name}.xlsx"
                for experiment in self.campaign_definition.experiments
            }
            for workbook_path in sorted(self.campaign_output_dir.glob("[0-9][0-9][0-9]_*/*.xlsx")):
                if workbook_path.name not in experiment_workbook_names:
                    continue
                try:
                    flux_frame = pd.read_excel(workbook_path, sheet_name="Fluxes", index_col=0)
                except ValueError:
                    continue
                if not flux_frame.empty:
                    flux_frames.append(flux_frame)
            fluxes = pd.concat(flux_frames, axis=1).sort_index() if flux_frames else None
            if fluxes is not None:
                fluxes.index.name = "Reaction ID"
            output_path = self.campaign_output_dir / f"{self.campaign_output_dir.name}_summary.xlsx"
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                if not summary.empty:
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                if fluxes is not None and not fluxes.empty:
                    fluxes.to_excel(writer, sheet_name="Fluxes")
            logger.info("Generated campaign Excel summary at '%s'.", output_path)

        logger.info("Campaign '%s' complete; recorded results for %d experiments.", self.campaign_definition.campaign_name, len(results))
        return results


    # ---------------------------------------------------------------------
    # Core runner
    # ---------------------------------------------------------------------
    def _run_single_experiment(
        self,
        experiment_def: structs.ExperimentDefinition,
        experiment_index: int,
        parallel_coordinates: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Runs one typed experiment definition.

        Args:
            experiment_def: Typed experiment definition to execute.
            experiment_index: Zero-based experiment index within the campaign.
            parallel_coordinates: Optional grid-point coordinates for
                execution modes that expand one experiment into multiple
                scheduler jobs.

        Returns:
            list[dict[str, Any]]: Payload rows returned by the experiment, or a
            failure payload if ``continue_on_error`` is enabled and execution
            fails.

        Raises:
            ValueError: If the experiment type or solver configuration cannot be
                resolved.
            NotImplementedError: If a recognized but unsupported solver mode is
                requested.
            TypeError: If the experiment returns a payload that is not a list.
            Exception: Re-raises the original experiment exception when
                ``continue_on_error`` is disabled.
        """
        # Every experiment gets a numbered output folder so filesystem order
        # matches campaign order. The filename cleaning is intentionally inline
        # here so the reader sees exactly how experiment names become paths.
        safe_output_name = "".join(char if (char.isalnum() or char in ("-", "_", " ")) else "_" for char in experiment_def.experiment_name).strip().replace(" ", "_")
        if not safe_output_name:
            safe_output_name = f"experiment_{experiment_index}"
        output_dir = self.campaign_output_dir / f"{experiment_index:03d}_{safe_output_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Prepared output directory '%s' for experiment '%s' (index=%d).", output_dir, experiment_def.experiment_name, experiment_index)

        started_at = datetime.now(timezone.utc)
        try:
            # Resolve the experiment type before solver construction so we can
            # reject invalid solver pairings before any heavy setup work.
            experiment_type = (experiment_def.experiment_type or getattr(experiment_def.analysis_config, "type", "") or "").strip().lower()
            experiment_class = self._EXPERIMENT_CLASS_MAP.get(experiment_type)
            if experiment_class is None:
                if not experiment_type: 
                    raise ValueError(f"Experiment '{experiment_def.experiment_name}' has no resolvable type. Set `experiment_type` or ensure `analysis_config.type` is present.")
                raise ValueError(f"Unsupported experiment type '{experiment_type}' for '{experiment_def.experiment_name}'. Supported: Characterization, SingleOptimization, DirectPostAnalyses, VariableScan, TwoVariableScan, FluxValidation, MCS.")

            # Solver construction stays inline so the launch path reads
            # top-to-bottom without bouncing through tiny helper methods.
            solver_kind = getattr(experiment_def.solver_config, "to_solve", "").lower()
            self._validate_solver_compatibility(
                experiment_type=experiment_type,
                solver_kind=solver_kind,
                experiment_name=experiment_def.experiment_name,
            )
            logger.debug("Experiment '%s' requested solver kind '%s'.", experiment_def.experiment_name, solver_kind)
            if solver_kind == "thermo":
                solver = ThermoSolverCPLEX(solver_config=experiment_def.solver_config)
            elif solver_kind == "stoichiometric":
                solver = StoichiometricSolver(solver_config=experiment_def.solver_config)
            elif solver_kind == "cobrak":
                solver = CobrakSolver(solver_config=experiment_def.solver_config)
            elif solver_kind == "mcs":
                solver = MCSSolverOPTLANG(solver_config=experiment_def.solver_config)
            elif solver_kind == "mcs_sd":
                solver = MCSSolverSD(solver_config=experiment_def.solver_config)
            elif solver_kind == "none":
                raise ValueError(f"Solver config for '{experiment_def.experiment_name}' is set to `to_solve='none'`, which is unsupported for experiment execution.")
            else:
                raise ValueError(f"Unknown solver configuration '{solver_kind}' for experiment '{experiment_def.experiment_name}'.")
            logger.debug("Instantiated solver '%s' for experiment '%s'.", type(solver).__name__, experiment_def.experiment_name)
            logger.info("Launching experiment '%s' with class='%s', solver='%s', output_dir='%s'.", experiment_def.experiment_name, experiment_class.__name__, type(solver).__name__, output_dir)

            # At this point the execution object gets the fully typed
            # experiment definition, the already-built solver, and the folder it
            # should use for any JSON/CSV artifacts it writes.
            experiment = experiment_class(experiment_def=experiment_def, solver=solver, output_dir=str(output_dir))

            # All experiment implementations are expected to normalize their
            # return value to a list of per-scenario dictionaries so campaign
            # bookkeeping stays uniform across experiment types.
            if parallel_coordinates is not None:
                payload = experiment.run(parallel_coordinates=parallel_coordinates)
            else:
                payload = experiment.run()
            if not isinstance(payload, list): 
                raise TypeError(f"Expected list payload from experiment run, got {type(payload).__name__}")
            logger.debug("Experiment '%s' returned %d payload rows.", experiment_def.experiment_name, len(payload))

            if parallel_coordinates is None:
                if experiment_def.analysis_config.analysis_params.json_output:
                    experiment.save_to_json()
                if experiment_def.analysis_config.analysis_params.csv_output:
                    experiment.save_to_csv()
                if experiment_def.analysis_config.analysis_params.excel_output:
                    experiment.save_to_excel()
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            logger.info("Experiment '%s' finished in %.2fs with %d scenario entries.", experiment_def.experiment_name, elapsed, len(payload))
            return payload

        except Exception as exc:
            logger.error("Experiment '%s' failed: %s", experiment_def.experiment_name, exc, exc_info=True)
            failure_payload = self._failure_payload(experiment_def=experiment_def, error=exc, started_at=started_at)
            self._write_failure_artifacts(output_dir, experiment_def.experiment_name, failure_payload)
            if not self.continue_on_error:
                raise
            logger.info("Continuing campaign after failure in experiment '%s' because continue_on_error=True.", experiment_def.experiment_name)
            return failure_payload

    def _save_campaign_snapshot(self) -> None:
        """Writes a JSON snapshot of the campaign plan to disk."""
        # The snapshot is a compact manifest of the campaign state before any
        # experiment runs. That makes later debugging easier because the exact
        # planned experiment order and scenario ids are preserved on disk.
        snapshot = {
            "campaign_name": self.campaign_definition.campaign_name,
            "output_dir": str(self.campaign_output_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_count": len(self.campaign_definition.experiments),
            "experiments": [
                {
                    "index": i,
                    "experiment_name": exp.experiment_name,
                    "experiment_type": exp.experiment_type,
                    "analysis_type": exp.analysis_config.type,
                    "n_scenarios": len(exp.biological_scenarios),
                    "scenario_ids": [s.scenario_id for s in exp.biological_scenarios],
                    "solver": type(exp.solver_config).__name__,
                }
                for i, exp in enumerate(self.campaign_definition.experiments)
            ],
        }

        snapshot_path = self.campaign_output_dir / "campaign_snapshot.json"
        logger.debug("Writing campaign snapshot for '%s' to '%s'.", self.campaign_definition.campaign_name, snapshot_path)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        logger.info("Saved campaign snapshot to '%s'.", snapshot_path)

    def _failure_payload(self, experiment_def: structs.ExperimentDefinition, error: BaseException, started_at: datetime) -> list[dict[str, Any]]:
        """Builds a standardized failure payload for one experiment.

        Args:
            experiment_def: Experiment definition that failed.
            error: Exception raised during execution.
            started_at: UTC timestamp captured at experiment start.

        Returns:
            list[dict[str, Any]]: Single-row payload describing the failure.
        """
        duration_s = (datetime.now(timezone.utc) - started_at).total_seconds()
        return [
            {
                "experiment_name": experiment_def.experiment_name,
                "scenario_id": "N/A",
                "substrate": "N/A",
                "product": "N/A",
                "status": f"Failed: {type(error).__name__}: {error}",
                "metrics": {"execution_time_seconds": round(duration_s, 4)},
                "steps": [],
                "post_analyses": {},
                "campaign_name": self.campaign_definition.campaign_name,
            }
        ]

    def _write_failure_artifacts(self, output_dir: Path, experiment_name: str, payload: list[dict[str, Any]]) -> None:
        """Writes failure payload artifacts for a failed experiment.

        Args:
            output_dir: Experiment output directory.
            experiment_name: Human-readable experiment name used to derive the
                artifact filename.
            payload: Failure payload to serialize as JSON.
        """
        # Failure filenames follow the same cleaning rule as experiment folders
        # so a reader can predict the artifact path without looking elsewhere.
        safe_experiment_name = "".join(char if (char.isalnum() or char in ("-", "_", " ")) else "_" for char in experiment_name).strip().replace(" ", "_")
        if not safe_experiment_name:
            safe_experiment_name = "experiment"
        error_path = output_dir / f"{safe_experiment_name}_failed.json"
        logger.debug("Writing failure artifact for experiment '%s' with %d payload rows to '%s'.", experiment_name, len(payload), error_path)
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved failure artifact to '%s'.", error_path)
