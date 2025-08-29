from __future__ import annotations

from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    NewType,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)

from gest.common.helpers.field_names import FieldNameMixin


class EntityProperties(FieldNameMixin):
    """Marker base-class for properties containers."""

    pass


class Gender(str, Enum):
    """String-based gender representation."""

    UNKNOWN = "unknown"
    MALE = "male"
    FEMALE = "female"


NodePrefix = NewType("NodePrefix", str)

ActorId = NewType("ActorId", str)
ObjectId = NewType("ObjectId", str)
LocationId = NewType("LocationId", str)
ActionId = NewType("ActionId", str)

NodeId = Union[ActorId, ObjectId, LocationId, ActionId]


ActionValue = NewType("ActionValue", str)
DEFAULT_ACTION_VALUE = ActionValue("Exists")


T = TypeVar("T", bound=EntityProperties)


class Entity(FieldNameMixin, Generic[T]):
    """Common fields shared by actors, objects, locations, and actions."""

    action: ActionValue = Field(
        default=DEFAULT_ACTION_VALUE,
        # TODO: Add description
        description="",
    )
    entities: List[NodeId] = Field(
        default_factory=list,
        # TODO: Add description
        description="",
    )
    location: Optional[List[NodeId]] = Field(
        default=None,
        # TODO: Add description
        description="",
    )
    timeframe: Optional[Any] = Field(
        default=None,
        # TODO: Add description
        description="",
    )
    properties: T = Field(
        # TODO: Add description
        description="",
    )


class BaseIdModel(BaseModel):
    """Base model for entities with an ID prefix."""

    id_prefix: ClassVar[NodePrefix]

    @classmethod
    def matches_id(cls, value: str) -> bool:
        """Check if a given value is a valid ID for this entity type."""

        p = cls.id_prefix
        return value.startswith(p) and value[len(p) :].isdigit()


class ActorProperties(EntityProperties):
    gender: Gender = Field(
        description=(
            "Actor's gender. "
            f"Allowed values: {', '.join(f'{e.value}' for e in Gender)}  "
        ),
    )
    name: str = Field(
        description="Human-readable label used in the narrative.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _trim_non_empty(cls, v: str) -> str:
        """Ensure that the name is non-empty and trimmed."""

        v = v.strip()
        if not v:
            raise ValueError("Actor name may not be empty.")
        return v


class Actor(BaseIdModel, Entity[ActorProperties]):
    id_prefix: ClassVar[NodePrefix] = NodePrefix("actor")

    # TODO: Ensure adding description for the other Entity fields

    properties: ActorProperties = Field(
        # TODO: Add description
        description="",
    )

    @field_validator("entities")
    @classmethod
    def _self_reference(cls, v: List[str]) -> List[str]:
        """Ensure that actors reference themselves in entities."""

        # Check that the actor references itself in entities
        if not v:
            raise ValueError("Actors must reference themselves in entities.")

        return v


class ObjectProperties(EntityProperties):
    type: str = Field(
        # TODO: Improve description
        description="Semantic class of the object.",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _trim_non_empty(cls, v: str) -> str:
        """Ensure that the object type is non-empty and trimmed."""

        v = v.strip()
        if not v:
            raise ValueError("Object type may not be empty.")
        return v


class Object(BaseIdModel, Entity[ObjectProperties]):
    id_prefix: ClassVar[NodePrefix] = NodePrefix("object")

    # TODO: Ensure adding description for the other Entity fields

    properties: ObjectProperties = Field(
        # TODO: Add description
        description="",
    )

    @field_validator("entities")
    @classmethod
    def _self_reference(cls, v: List[str]) -> List[str]:
        """Ensure that objects reference themselves in entities."""

        # Check that the object references itself in entities
        if not v:
            raise ValueError("Objects must reference themselves in entities.")

        return v


class LocationProperties(EntityProperties):
    type: str = Field(
        # TODO: Improve description
        description="Semantic class of the location.",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _trim_non_empty(cls, v: str) -> str:
        """Ensure that the location type is non-empty and trimmed."""

        v = v.strip()
        if not v:
            raise ValueError("Location type may not be empty.")
        return v


class Location(BaseIdModel, Entity[LocationProperties]):
    id_prefix: ClassVar[NodePrefix] = NodePrefix("location")

    # TODO: Ensure adding description for the other Entity fields

    properties: LocationProperties = Field(
        # TODO: Add description
        description="",
    )

    @field_validator("entities")
    @classmethod
    def _self_reference(cls, v: List[str]) -> List[str]:
        """Ensure that locations reference themselves in entities."""

        # Check that the location references itself in entities
        if not v:
            raise ValueError("Locations must reference themselves in entities.")

        return v


class ActionProperties(EntityProperties):
    pass


class Action(BaseIdModel, Entity[ActionProperties]):
    """Represents a verb-centric event involving one or more entities."""

    id_prefix: ClassVar[NodePrefix] = NodePrefix("action")

    # TODO: Ensure adding description for the other Entity fields

    action: ActionValue = Field(
        default_factory=lambda: ActionValue(""),
        # TODO: Improve description
        description="Atomic verb performed (non-empty and **not** 'Exists')",
    )
    properties: ActionProperties = Field(
        # TODO: Add description
        description="",
    )

    @field_validator("action")
    @classmethod
    def _verb_required(cls, v: ActionValue) -> ActionValue:
        """Ensure that the action has a non-empty verb and is not the default value."""

        # Check that the verb is not empty
        if not v.strip():
            raise ValueError("Every Action must have a non-empty verb.")

        # Check that the verb is not the default value
        if v == DEFAULT_ACTION_VALUE:
            raise ValueError(
                f"'{DEFAULT_ACTION_VALUE}' is reserved for entity existence, not action verbs."
            )

        return v

    @field_validator("entities")
    @classmethod
    def _has_entities(cls, v: List[str]) -> List[str]:
        """Ensure that the action has at least one entity and that they are valid."""

        # Check that at least one entity is provided
        if not v:
            raise ValueError("Action requires at least one entity.")

        # Check that all entities are valid Actor, Object or Location IDs
        invalid_ids = [
            eid
            for eid in v
            if not (
                Actor.matches_id(eid)
                or Object.matches_id(eid)
                or Location.matches_id(eid)
            )
        ]
        if invalid_ids:
            raise ValueError(
                f"Entities must only reference actors, objects, or locations. Invalid IDs: {invalid_ids}"
            )

        return v


class TimelineEntry(FieldNameMixin):
    """Represents a single entry in an actor's timeline."""

    action_id: ActionId = Field(
        # TODO: Improve description
        description="Action ID referenced in this timeline slot.",
    )
    verb: ActionValue = Field(
        # TODO: Improve description
        description="The verb value from the referenced action.",
    )


TemporalMarkerId = NewType("TemporalMarkerId", str)


class AllenRelation(str, Enum):
    """
    Allen's interval-algebra relations
    """

    # Defined relations (a ⟂ b):

    # a BEFORE b          ───a────   ───b────
    # a AFTER b           ───b────   ───a────
    # a MEETS b           ───a────┐┌───b────
    # a MET_BY b          ───b────┐┌───a────
    # a OVERLAPS b        ──a───┐     ┌──b───
    # a OVERLAPPED_BY b   ──b───┐     ┌──a───
    # a STARTS b          ───a───┐     └─b─┘
    # a STARTED_BY b      ───b───┐     └─a─┘
    # a FINISHES b        ┌─a─┐     ┌───b───┘
    # a FINISHED_BY b     ┌─b─┐     ┌───a───┘
    # a DURING b          ┌─a─┐     ┌────b────┐
    # a CONTAINS b        ┌────a────┐┌─b─┐
    # a EQUAL b           ┌───a,b───┐

    # The original relations
    BEFORE = "before"
    MEETS = "meets"
    OVERLAPS = "overlaps"
    STARTS = "starts"
    DURING = "during"
    FINISHES = "finishes"
    EQUAL = "equal"

    # Inverse relations
    AFTER = "after"
    MET_BY = "met_by"
    OVERLAPPED_BY = "overlapped_by"
    STARTED_BY = "started_by"
    CONTAINS = "contains"
    FINISHED_BY = "finished_by"


class TemporalMarker(BaseIdModel, FieldNameMixin):
    """Represents a temporal marker that anchors an action to a target action."""

    id_prefix: ClassVar[NodePrefix] = NodePrefix("tm")

    type: AllenRelation = Field(
        description=(
            "Allen interval-algebra relation describing how two actions are temporally linked.  "
            f"Allowed values: {', '.join(f'{e.value}' for e in AllenRelation)}."
        )
    )


class TemporalRelation(BaseModel):
    """Represents a temporal relation between two actions using a temporal marker."""

    type: TemporalMarkerId = Field(
        # TODO: Improve description
        description="ID of a TemporalMarker that specifies the Allen relation.",
    )
    target: ActionId = Field(
        # TODO: Improve description
        description="The Action that is related to the source Action.",
    )

    @field_validator("target")
    @classmethod
    def _target_is_action_id(cls, v: str) -> str:
        """Ensure that 'target' looks like an ActionId."""
        if not Action.matches_id(v):
            raise ValueError(
                f"TemporalRelation's 'target' must be an ActionId, got '{v}'."
            )
        return v

    @field_validator("type")
    @classmethod
    def _type_is_marker_id(cls, v: str) -> str:
        """Ensure that 'type' looks like a TemporalMarkerId."""
        if not TemporalMarker.matches_id(v):
            raise ValueError(
                f"TemporalRelation's 'type' must be a TemporalMarkerId, got '{v}'."
            )
        return v


class TemporalAction(FieldNameMixin):
    """Represents temporal metadata for an action, including relations and next action."""

    relations: List[TemporalRelation] = Field(
        default_factory=list,
        # TODO: Add description
        description="",
    )
    next: List[ActionId] = Field(
        default_factory=list,
        # TODO: Add description
        description="",
    )

    @field_validator("next")
    @classmethod
    def _next_are_action_ids(cls, v: List[str]) -> List[str]:
        """Ensure that 'next' contains only valid ActionIds."""
        invalid = [x for x in v if not Action.matches_id(x)]
        if invalid:
            raise ValueError(
                f"TemporalAction's 'next' contains non-ActionIds: {invalid}"
            )
        return v


RawTimeline: TypeAlias = Dict[str, Any]

Timeline: TypeAlias = Dict[ActorId, List[TimelineEntry]]
StartingActions: TypeAlias = Dict[ActorId, ActionId]
TemporalActions: TypeAlias = Dict[ActionId, TemporalAction]
TemporalMarkers: TypeAlias = Dict[TemporalMarkerId, TemporalMarker]


class Temporal(FieldNameMixin):
    """Narrative ordering of actions along individual actor timelines."""

    timeline: Timeline = Field(
        default_factory=dict,
        # TODO: Improve description
        description=(
            "Maps each actor ID to the ordered list of TimelineEntry pair. "
            "This tells the actor's story."
        ),
    )
    starting_actions: StartingActions = Field(
        default_factory=dict,
        # TODO: Improve description
        description="First action of every actor's timeline (anchor point).",
    )
    actions: TemporalActions = Field(
        default_factory=dict,
        # TODO: Improve description
        description=(
            "Per-action temporal metadata - `relations` (links via TemporalMarkers) and `next` (immediate successor action)."
        ),
    )
    temporal_markers: TemporalMarkers = Field(
        default_factory=dict,
        # TODO: Improve description
        description="Dictionary of TemporalMarker instances keyed by their IDs.",
    )

    @staticmethod
    def _encode_timeline(tl: Timeline) -> RawTimeline:
        """Encode a timeline into a raw dictionary format."""

        return {
            actor: [{"action_id": e.action_id, "verb": e.verb} for e in entries]
            for actor, entries in tl.items()
        }

    @staticmethod
    def _decode_timeline(raw: RawTimeline) -> Timeline:
        """Decode a raw timeline dictionary into a structured Timeline object."""

        out: Timeline = {}

        for actor, seq in raw.items():
            entries: list[TimelineEntry] = []
            for item in seq:
                if isinstance(item, dict):
                    a = item["action_id"]
                    v = item["verb"]
                else:
                    a, v = item

                entries.append(TimelineEntry(action_id=a, verb=ActionValue(v)))

            out[ActorId(actor)] = entries

        return out


SpatialId = Union[ActorId, ObjectId, LocationId]

SpatialMarkerId = NewType("SpatialMarkerId", str)

SpatialDetail = NewType("SpatialDetail", str)


class SpatialRelation(BaseModel):
    """Represents a spatial relation between two actions using a spatial marker."""

    type: SpatialMarkerId = Field(
        # TODO: Improve description
        description="ID of a SpatialMarker that specifies the spatial relation.",
    )
    target: SpatialId = Field(
        # TODO: Improve description
        description="The Actor, Object, or Location that is related to the source Action.",
    )
    detail: Optional[SpatialDetail] = Field(
        default=None,
        # TODO: Improve description
        description="Additional details about the spatial relation.",
    )

    @field_validator("target")
    @classmethod
    def _target_is_spatial_id(cls, v: str) -> str:
        """Ensure that 'target' is an ActorId, ObjectId, or LocationId."""
        if not (Actor.matches_id(v) or Object.matches_id(v) or Location.matches_id(v)):
            raise ValueError(
                f"SpatialRelation's 'target' must be Actor/Object/Location id, got '{v}'."
            )
        return v

    @field_validator("type")
    @classmethod
    def _type_is_spatial_marker_id(cls, v: str) -> str:
        """Ensure that 'type' looks like a SpatialMarkerId."""
        if not SpatialMarker.matches_id(v):
            raise ValueError(
                f"SpatialRelation's 'type' must be a SpatialMarkerId, got '{v}'."
            )
        return v

    @field_validator("detail", mode="before")
    @classmethod
    def _trim_non_empty(cls, v: Optional[str]) -> Optional[str]:
        """Ensure that the spatial relation detail is non-empty and trimmed."""

        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Spatial relation detail may not be empty.")
        return v


QualitativeSpatialReasoning = NewType("QualitativeSpatialReasoning", str)


class SpatialMarker(BaseIdModel, FieldNameMixin):
    """Represents a spatial marker that anchors an action to a target entity."""

    id_prefix: ClassVar[NodePrefix] = NodePrefix("sm")

    type: QualitativeSpatialReasoning = Field(
        # TODO: Add description
        description="",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _trim_non_empty(cls, v: str) -> str:
        """Ensure that the spatial marker type is non-empty and trimmed."""

        v = v.strip()
        if not v:
            raise ValueError("SpatialMarker type may not be empty.")
        return v


class SpatialAction(FieldNameMixin):
    """Represents spatial metadata for an action, including spatial relations."""

    relations: List[SpatialRelation] = Field(
        default_factory=list,
        # TODO: Improve description
        description="List of spatial relations (marker id + target entity).",
    )


SpatialActions: TypeAlias = Dict[ActionId, SpatialAction]
SpatialMarkers: TypeAlias = Dict[SpatialMarkerId, SpatialMarker]


class Spatial(FieldNameMixin):
    """Represents spatial metadata for actions, including markers and relations."""

    actions: SpatialActions = Field(
        default_factory=dict,
        # TODO: Improve description
        description="Per-action spatial metadata.",
    )
    spatial_markers: SpatialMarkers = Field(
        default_factory=dict,
        # TODO: Improve description
        description="All SpatialMarker objects keyed by ID.",
    )


RawGEST: TypeAlias = Dict[str, Any]

ActorEntity: TypeAlias = Dict[ActorId, Actor]
ObjectEntity: TypeAlias = Dict[ObjectId, Object]
LocationEntity: TypeAlias = Dict[LocationId, Location]
ActionEntity: TypeAlias = Dict[ActionId, Action]


class GEST(FieldNameMixin):
    """Graph of Events in Space and Time."""

    actors: ActorEntity = Field(
        default_factory=dict,
        # TODO: Improve description
        description="All Actor entities keyed by their IDs.",
    )
    objects: ObjectEntity = Field(
        default_factory=dict,
        # TODO: Improve description
        description="All Object entities keyed by their IDs.",
    )
    locations: LocationEntity = Field(
        default_factory=dict,
        # TODO: Improve description
        description="All Location entities keyed by their IDs.",
    )
    actions: ActionEntity = Field(
        default_factory=dict,
        # TODO: Improve description
        description="All Action entities keyed by their IDs.",
    )
    temporal: Temporal = Field(
        default_factory=Temporal,
        # TODO: Improve description
        description="Temporal layer (when actions happen).",
    )
    spatial: Spatial = Field(
        default_factory=Spatial,
        # TODO: Improve description
        description="Spatial layer (where actions happen).",
    )

    @property
    def all_entities(self) -> Dict[NodeId, Entity]:
        """Unified mapping of every Entity in the graph."""

        out: Dict[NodeId, Entity] = {}
        for mapping in (self.actors, self.objects, self.locations, self.actions):
            out.update(mapping.items())

        return out

    @model_validator(mode="before")
    @classmethod
    def _inflate_flat(cls, data: RawGEST) -> RawGEST:
        """Inflate a flat GEST dictionary into a structured GEST object."""

        g = cls.fields()
        t = Temporal.fields()
        s = Spatial.fields()

        if not isinstance(data, Dict):
            raise ValueError(
                f"Expected a dictionary for GEST data, got {type(data).__name__}."
            )

        if g.temporal not in data:
            raise ValueError(f"Missing required key '{g.temporal}' in GEST data.")

        if g.spatial not in data:
            raise ValueError(f"Missing required key '{g.spatial}' in GEST data.")

        return {
            g.actors: {k: data[k] for k in data if Actor.matches_id(k)},
            g.objects: {k: data[k] for k in data if Object.matches_id(k)},
            g.locations: {k: data[k] for k in data if Location.matches_id(k)},
            g.actions: {
                k: data[k] for k in data if Action.matches_id(k) and k != g.temporal
            },
            g.temporal: {
                t.timeline: Temporal._decode_timeline(data[g.temporal][t.timeline]),
                t.starting_actions: data[g.temporal].get(t.starting_actions, {}),
                t.actions: {
                    k: v for k, v in data[g.temporal].items() if Action.matches_id(k)
                },
                t.temporal_markers: {
                    k: v
                    for k, v in data[g.temporal].items()
                    if TemporalMarker.matches_id(k)
                },
            },
            g.spatial: {
                s.actions: {
                    k: v for k, v in data[g.spatial].items() if Action.matches_id(k)
                },
                s.spatial_markers: {
                    k: v
                    for k, v in data[g.spatial].items()
                    if SpatialMarker.matches_id(k)
                },
            },
        }

    @model_serializer(mode="wrap")
    def _flatten(self, ser: Callable) -> RawGEST:
        """Flatten the GEST into a dictionary representation."""

        nested: RawGEST = ser(self)

        g = self.fields()
        t = Temporal.fields()
        ta = TemporalAction.fields()
        tm = TemporalMarker.fields()
        e_fields = TimelineEntry.fields()
        s = Spatial.fields()
        sa = SpatialAction.fields()

        flat: RawGEST = {
            **nested[g.actors],
            **nested[g.objects],
            **nested[g.locations],
            **nested[g.actions],
            g.temporal: {},
            g.spatial: {},
        }

        # Flatten the temporal layer
        tl_dict = nested[g.temporal][t.timeline]
        flat[g.temporal][t.timeline] = {
            actor: [[e[e_fields.action_id], e[e_fields.verb]] for e in entries]
            for actor, entries in tl_dict.items()
        }

        flat[g.temporal][t.starting_actions] = nested[g.temporal][t.starting_actions]

        for meta_id, meta in nested[g.temporal][t.actions].items():
            flat[g.temporal][meta_id] = {
                k: v
                for k, v in meta.items()
                if (k in {ta.relations, ta.next}) or v is not None
            }

        for meta_id, meta in nested[g.temporal][t.temporal_markers].items():
            flat[g.temporal][meta_id] = {
                k: v for k, v in meta.items() if (k in {tm.type}) or v is not None
            }

        # Flatten the spatial layer
        for meta_id, meta in nested[g.spatial][s.actions].items():
            flat[g.spatial][meta_id] = {
                k: v for k, v in meta.items() if k == sa.relations and v
            }

        for meta_id, marker in nested[g.spatial][s.spatial_markers].items():
            flat[g.spatial][meta_id] = marker

        return flat

    @model_validator(mode="after")
    def _cross_check(self) -> GEST:
        """Perform cross-validation checks on the GEST data."""

        # Check that all actors reference themselves in Entities
        entity_ids = set(self.actors) | set(self.objects) | set(self.locations)
        for act_id, act in self.actions.items():
            missing = set(act.entities) - entity_ids
            if missing:
                raise ValueError(f"{act_id} references unknown entities: {missing}")

        # Check that all actions in the temporal metadata have corresponding Action entries
        declared, timed = set(self.actions), set(self.temporal.actions)
        if lost := declared - timed:
            raise ValueError(f"Missing temporal metadata for actions: {lost}")

        # Check that all actions in the timeline have corresponding Action entries
        for actor_id, entries in self.temporal.timeline.items():
            for entry in entries:
                action = self.actions.get(entry.action_id)
                # Check that the action exists in the GEST
                if action is None:
                    raise ValueError(
                        f"Timeline for {actor_id} references unknown action {entry.action_id}"
                    )

                # Check that the verb matches the action's verb
                if entry.verb != action.action:
                    raise ValueError(
                        f"Verb mismatch for {entry.action_id}: timeline says '{entry.verb}', but Action.action is '{action.action}'"
                    )

        # Check that all temporal actions have valid relations and next actions
        for ta_id, ta in self.temporal.actions.items():
            # Check that the relations are well-formed
            if ta.relations:
                for rel in ta.relations:
                    # Check that the relation type is a known temporal marker
                    if rel.type not in self.temporal.temporal_markers:
                        raise ValueError(
                            f"{ta_id} references unknown temporal marker '{rel.type}'."
                        )

                    # Check that the target action is known
                    if rel.target not in self.actions:
                        raise ValueError(
                            f"{ta_id} has a relation pointing to unknown action '{rel.target}'."
                        )

                    # Check that the marker has a valid type
                    marker = self.temporal.temporal_markers[rel.type]
                    if marker.type is None:
                        raise ValueError(
                            f"Temporal marker {rel.type} (used in {ta_id}) has no 'type'."
                        )

            # Check that the next actions are well-formed
            for nxt in ta.next:
                # Check that the 'next' action is known
                if nxt not in self.actions:
                    raise ValueError(f"{ta_id}.next points to unknown action '{nxt}'.")

                # Check that the 'next' action is not the same as the source action
                if nxt == ta_id:
                    raise ValueError(f"{ta_id}.next must reference a different action.")

        # Check for temporal cycles in the actions graph
        def _dfs(aid: ActionId, visiting: set[str], visited: set[str]) -> None:
            if aid in visiting:
                raise ValueError(f"Temporal cycle detected at {aid}.")

            if aid in visited:
                return

            visiting.add(aid)

            for nxt in self.temporal.actions[aid].next:
                _dfs(nxt, visiting, visited)

            visiting.remove(aid)
            visited.add(aid)

        visited: set[str] = set()
        for root in self.temporal.starting_actions.values():
            _dfs(root, set(), visited)

        # Check that the timeline order is mirrored in TemporalAction.next
        for actor_id, seq in self.temporal.timeline.items():
            for cur, nxt in zip(seq, seq[1:]):
                if nxt.action_id not in self.temporal.actions[cur.action_id].next:
                    raise ValueError(
                        f"Timeline order {cur.action_id} → {nxt.action_id} for {actor_id} not mirrored in TemporalAction.next."
                    )

        # Check that all temporal markers are used in relations
        unused_tm = set(self.temporal.temporal_markers)
        for ta in self.temporal.actions.values():
            unused_tm.difference_update(rel.type for rel in ta.relations)

        if unused_tm:
            raise ValueError(
                f"Temporal markers defined, but never used: {sorted(unused_tm)}"
            )

        # Check that all spatial actions have valid relations and targets
        for sa_id, sa in self.spatial.actions.items():
            for rel in sa.relations:
                # Check that the relation type is a known spatial marker
                if rel.type not in self.spatial.spatial_markers:
                    raise ValueError(
                        f"{sa_id} references unknown spatial marker '{rel.type}'."
                    )

                # Check that the target is a valid SpatialId
                if (
                    rel.target not in self.all_entities
                    and rel.target not in self.actions
                ):
                    raise ValueError(
                        f"{sa_id} has a spatial relation pointing to unknown id '{rel.target}'."
                    )

                # Check that the marker has a valid type
                marker = self.spatial.spatial_markers[rel.type]
                if marker.type is None:
                    raise ValueError(
                        f"Spatial marker {rel.type} (used in {sa_id}) has no 'type'."
                    )

        # Check that all spatial markers are used in relations
        unused_sm = set(self.spatial.spatial_markers)
        for sa in self.spatial.actions.values():
            unused_sm.difference_update(rel.type for rel in sa.relations)

        if unused_sm:
            raise ValueError(
                f"Spatial markers defined, but never used: {sorted(unused_sm)}"
            )

        return self

    @staticmethod
    def _patch_root_schema(schema: Dict[str, Any]) -> None:
        """Patches the root schema by removing collection keys and adding pattern properties."""

        ## Flatten 'actors', 'objects', 'locations' and 'actions'
        for k in ("actors", "objects", "locations", "actions"):
            schema["properties"].pop(k, None)

        schema.setdefault("patternProperties", {}).update(
            {
                r"^actor\d+$": {"$ref": "#/$defs/Actor"},
                r"^object\d+$": {"$ref": "#/$defs/Object"},
                r"^location\d+$": {"$ref": "#/$defs/Location"},
                r"^action\d+$": {"$ref": "#/$defs/Action"},
            }
        )
        schema["additionalProperties"] = False

        ## Flatten 'temporal'
        t = schema["$defs"]["Temporal"]
        for k in ("actions", "temporal_markers"):
            t["properties"].pop(k, None)

        t.setdefault("patternProperties", {}).update(
            {
                r"^action\d+$": {"$ref": "#/$defs/TemporalAction"},
                r"^tm\d+$": {"$ref": "#/$defs/TemporalMarker"},
            }
        )
        t["additionalProperties"] = False

        ## Flatten 'spatial'
        s = schema["$defs"]["Spatial"]
        for k in ("actions", "spatial_markers"):
            s["properties"].pop(k, None)

        s.setdefault("patternProperties", {}).update(
            {
                r"^action\d+$": {"$ref": "#/$defs/SpatialAction"},
                r"^sm\d+$": {"$ref": "#/$defs/SpatialMarker"},
            }
        )
        s["additionalProperties"] = False

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        """Overrides the model JSON schema for GEST."""

        schema = super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
        )
        cls._patch_root_schema(schema)
        return schema
