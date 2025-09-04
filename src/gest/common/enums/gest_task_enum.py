from enum import Enum


class GestTaskEnum(str, Enum):
    """Supported types of GEST tasks."""

    GENERATION = "generation"
    EVALUATION = "evaluation"
