from gest.data.gest import GEST


class BaseGESTException(Exception):
    """Base GEST exception class. All GEST exceptions should inherit from this class."""


class GESTGenerationError(BaseGESTException):
    """GEST generation error class. Raised to catch excepted problems during GEST generation phase."""


class GESTContentSimilarityError(BaseGESTException):
    """GEST content similarity error class. Raised to catch excepted problems during GEST content similarity evaluation improvement phase."""

    def __init__(
        self,
        message: str,
        original_user_input: str,
        proposed_gest_content: GEST,
        recomputed_narrative: str,
        current_text_similarity_score: float,
    ):
        self.message = message
        self.original_user_input = original_user_input
        self.proposed_gest_content = proposed_gest_content
        self.recomputed_narrative = recomputed_narrative
        self.current_text_similarity_score = current_text_similarity_score

        super().__init__(self.message)


class GESTValueError(BaseGESTException):
    """GEST value error class. Raised to catch excepted problems during GEST value evaluation improvement phase."""
