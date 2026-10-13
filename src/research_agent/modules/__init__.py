"""Scientific module entry points."""

from .gap_discovery import GapDiscoveryConfig, GapDiscoveryModule
from .solution_ideation import SolutionIdeationConfig, SolutionIdeationModule
from .related_work_verification import (
    RelatedWorkVerificationModule,
    VerificationConfig,
)
from .inspiration_retrieval import (
    InspirationRetrievalConfig,
    InspirationRetrievalModule,
)
from .targeted_revision import (
    TargetedRevisionConfig,
    TargetedRevisionExecutionResult,
    TargetedRevisionModule,
    apply_revision,
    canonicalize_revision_result,
)
from .experiment_design import (
    ExperimentDesignConfig,
    ExperimentDesignExecutionResult,
    ExperimentDesignModule,
)
from .commit import (
    CommitModule,
    check_commit_gate,
    create_commit_snapshot,
    load_commit_snapshot,
    save_commit_snapshot,
)
from .finalizer import (
    FinalizerConfig,
    FinalizerModule,
    load_final_research_proposal,
    save_final_research_proposal,
)

from .seed_context import (
    SeedContextConfig,
    SeedContextExtractor,
    write_seed_context_outputs,
)

__all__ = [
    "SeedContextConfig",
    "SeedContextExtractor",
    "write_seed_context_outputs",
    "GapDiscoveryConfig",
    "GapDiscoveryModule",
    "SolutionIdeationConfig",
    "SolutionIdeationModule",
    "RelatedWorkVerificationModule",
    "VerificationConfig",
    "InspirationRetrievalConfig",
    "InspirationRetrievalModule",
    "TargetedRevisionConfig",
    "TargetedRevisionExecutionResult",
    "TargetedRevisionModule",
    "apply_revision",
    "canonicalize_revision_result",
    "ExperimentDesignConfig",
    "ExperimentDesignExecutionResult",
    "ExperimentDesignModule",
    "CommitModule",
    "check_commit_gate",
    "create_commit_snapshot",
    "load_commit_snapshot",
    "save_commit_snapshot",
    "FinalizerConfig",
    "FinalizerModule",
    "load_final_research_proposal",
    "save_final_research_proposal",
]
