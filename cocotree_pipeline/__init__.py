from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import COCOTreeGenerator
    from .types import GeneratorConfig, NodeInfo, RunResult

__all__ = ["COCOTreeGenerator", "GeneratorConfig", "NodeInfo", "RunResult"]


def __getattr__(name: str):
    if name == "COCOTreeGenerator":
        from .generator import COCOTreeGenerator

        return COCOTreeGenerator
    if name in {"GeneratorConfig", "NodeInfo", "RunResult"}:
        from . import types

        return getattr(types, name)
    raise AttributeError(name)
