from enum import Enum


class FlowOperationNameEnum(str, Enum):
    """
    Enum for flow operation names used in GEST generation and improvement.
    This enum is used to identify the type of operation being performed.
    """

    GENERATION = "generation"
    IMPROVEMENT = "improvement"
