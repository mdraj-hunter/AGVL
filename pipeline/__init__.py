"""AGVL pipeline orchestration."""

__all__ = ["run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .agvl_pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(name)
