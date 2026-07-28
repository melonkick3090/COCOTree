from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROOT_INSTANCE_ID = "ROOT"
ROOT_SEMANTIC_ID = "ROOT"


@dataclass(frozen=True)
class InstanceNode:
    """One scored mask instance and its semantic/instance-tree links."""

    image_id: int
    instance_id: str
    semantic_id: str
    label: str
    parent_instance_id: str
    parent_semantic_id: str
    segmentation: dict[str, Any]
    semantic_depth: int = 0


@dataclass
class ImageTree:
    image_id: int
    nodes: list[InstanceNode] = field(default_factory=list)

    def sorted_nodes(self) -> list[InstanceNode]:
        return sorted(self.nodes, key=lambda node: natural_id_key(node.instance_id))

    @property
    def instance_ids(self) -> set[str]:
        return {node.instance_id for node in self.nodes}

    @property
    def semantic_ids(self) -> set[str]:
        return {node.semantic_id for node in self.nodes}


def natural_id_key(value: str) -> tuple[str, int, str]:
    text = str(value)
    prefix = text.rstrip("0123456789")
    suffix = text[len(prefix):]
    return (prefix, int(suffix) if suffix else -1, text)

