from __future__ import annotations

from typing import Any, Literal, Protocol

from mymodelingpackage.structures import BiologicalScenario, SolverResult


class BiologicalSolver(Protocol):
    """Protocol for stoichiometric, thermodynamic, and COBRA-k solvers."""

    config: Any
    solver_config: Any

    def load_context(self, biological_scenario: BiologicalScenario, analysis_config: Any) -> None:
        """Load the biological scenario and analysis configuration before solve()."""
        ...

    def supports(self, capability: str) -> bool:
        """Report optional solver capabilities used by dispatchers and workflows."""
        ...

    def solve(
        self,
        objective_var_ids: list[str],
        objective_multipliers: list[float] | None = None,
        objective_direction: Literal["max", "min"] = "max",
        constraints: list[list[Any]] | None = None,
        warm_start_solution: dict[str, Any] | None = None,
        subnetwork_ids: list[str] | None = None,
        thermo_unconstrained_reactions: list[str] | None = None,
    ) -> SolverResult:
        """Execute a flux/thermo optimization and return objective/variable values."""
        ...


