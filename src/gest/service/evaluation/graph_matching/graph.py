import json
from functools import cached_property
from typing import Any, Iterable, List, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict

from gest.data.gest import (
    GEST,
    ActionId,
    ActionValue,
    AllenRelation,
    Entity,
    NodeId,
    QualitativeSpatialReasoning,
    Spatial,
    SpatialAction,
    SpatialMarker,
    SpatialMarkerId,
    Temporal,
    TemporalAction,
    TemporalMarker,
    TemporalMarkerId,
)

__all__ = [
    "AllenRelation",
    "QualitativeSpatialReasoning",
    "SpatialAction",
    "SpatialMarker",
    "SpatialMarkerId",
    "TemporalAction",
    "TemporalMarker",
    "TemporalMarkerId",
]


class TimelineEdge(NamedTuple):
    """Represents a directed edge in the timeline graph."""

    src: ActionId
    dst: ActionId


class GESTGraph(BaseModel):
    """A graph representation of a GEST (Graph of Events in Space and Time)."""

    gest: GEST

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def __hash__(self) -> int:
        return hash(
            json.dumps(
                self.model_dump(exclude_none=True, by_alias=True), sort_keys=True
            )
        )

    def action(self, node_id: NodeId) -> ActionValue:
        """Get the action associated with a node."""

        node = self._get_node(node_id)

        if not isinstance(node, Entity):
            raise KeyError(
                f"Node '{node_id}' not found in GEST or does not have an action."
            )

        return node.action

    def entities(self, node_id: NodeId) -> List[NodeId]:
        """Get the entities associated with a node."""

        node = self._get_node(node_id)

        if not isinstance(node, Entity):
            raise KeyError(f"Node {node_id!r} is not an Entity.")

        return node.entities

    def timeframe(self, node_id: NodeId) -> Optional[Any]:
        """Get the timeframe associated with a node."""

        node = self._get_node(node_id)

        if not isinstance(node, Entity):
            raise KeyError(
                f"Node '{node_id}' not found in GEST or does not have a timeframe."
            )

        return node.timeframe

    def location(self, node_id: NodeId) -> Optional[List[NodeId]]:
        """Get the location associated with a node."""

        node = self._get_node(node_id)

        if not isinstance(node, Entity):
            raise KeyError(
                f"Node '{node_id}' not found in GEST or does not have a location."
            )

        return node.location

    @property
    def action_nodes(self) -> Iterable[ActionId]:
        """Get all action nodes in the GEST."""

        return list(self.gest.actions.keys())

    @property
    def temporal(self) -> Temporal:
        """Get the temporal representation of the GEST."""

        return self.gest.temporal

    @property
    def spatial(self) -> Spatial:
        """Get the spatial representation of the GEST."""

        return self.gest.spatial

    @cached_property
    def _node_index(self) -> dict[NodeId, Entity]:
        """Create an index of all nodes in the GEST for quick access."""

        return self.gest.all_entities

    def _get_node(self, node_id: NodeId, *, strict: bool = True) -> Optional[Entity]:
        """
        Return the node with *node_id* or *None*.

        If *strict* is True, raise KeyError when the id is absent.
        """

        node = self._node_index.get(node_id)

        if node is None and strict:
            raise KeyError(f"Node '{node_id}' not found in GEST.")

        return node
