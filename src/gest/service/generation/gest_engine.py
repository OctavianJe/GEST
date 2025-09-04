import json
import logging
import time
from typing import Callable, Optional, cast

from pydantic import BaseModel, PrivateAttr, ValidationError

from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError
from gest.common.helpers.jinja_helper import JinjaHelper
from gest.common.helpers.string_helper import StringHelper
from gest.data.gest import GEST
from gest.service.generation.enums.flow_operation_name_enum import FlowOperationNameEnum
from gest.service.generation.exception.gest_exceptions import (
    GESTContentSimilarityError,
    GESTGenerationError,
    GESTValueError,
)
from gest.service.other.llm.provider.base_llm_provider import (
    BaseLLMProvider,
    LLMExpectedResultType,
    LLMProviderEnum,
)
from gest.service.other.llm.provider.exception.provider_response_exceptions import (
    BaseLLMProviderResponseError,
)
from gest.service.other.llm.provider.provider_factory import LLMProviderFactory
from gest.service.other.text_similarity.text_similarity_evaluator import (
    TextSimilarityEvaluator,
)


class GESTEngine(BaseModel):
    """GEST Engine for generating GEST objects from content using LLM providers."""

    _jinja_helper: JinjaHelper

    # Engine configs
    _compute_retries: int = PrivateAttr()
    _compute_delay: int = PrivateAttr()

    # Generation configs
    _generation_llm_provider_enum: LLMProviderEnum = PrivateAttr()
    _generation_retries: int = PrivateAttr()
    _generation_delay: int = PrivateAttr()

    # Improvement configs
    _improvement_llm_provider_enum: LLMProviderEnum = PrivateAttr()
    _improvement_retries: int = PrivateAttr()
    _improvement_delay: int = PrivateAttr()
    _improvement_text_similarity_threshold: float = PrivateAttr()
    _improvement_include_exact_no_sentences: bool = PrivateAttr()

    def __init__(self):
        self._assign_configs()

        self._jinja_helper = JinjaHelper()

    def compute(self, content: str) -> GEST:
        """
        Computes a GEST object from the given content.

        This method orchestrates the generation and improvement process, with a retry
        mechanism that re-runs the entire flow if the improved GEST fails the
        content similarity check.
        """

        generate_llm_provider = LLMProviderFactory().create_provider(
            type=self._generation_llm_provider_enum
        )

        minimum_text_similarity_score = 0.0
        maximum_text_similarity_score = 0.0

        for attempt in range(1, self._compute_retries + 1):
            try:
                logging.info(
                    f"Starting GEST generation process, attempt {attempt}/{self._compute_retries}."
                )
                # Step 1: Generate the initial GEST object
                initial_gest = self._generate(
                    llm_provider=generate_llm_provider,
                    content=content,
                )

                if self._improvement_skip_step:
                    logging.info(
                        "Skipping improvement step is enabled, either if not recommended unless ablation study."
                    )
                    return initial_gest

                # Step 2: Improve the GEST object
                improved_gest = self._improve(
                    generated_gest=initial_gest,
                    original_content=content,
                )

                # If both steps succeed, return the result
                return improved_gest

            except GESTContentSimilarityError as err:
                current_score = err.current_text_similarity_score
                logging.warning(
                    f"Attempt {attempt} failed: Content similarity check failed. "
                    f"Current similarity score is {current_score}. "
                    f"Error: '{err}'."
                )

                # Update minimum and maximum similarity scores across attempts
                if attempt == 1:
                    minimum_text_similarity_score = current_score
                    maximum_text_similarity_score = current_score
                else:
                    minimum_text_similarity_score = min(
                        minimum_text_similarity_score, current_score
                    )
                    maximum_text_similarity_score = max(
                        maximum_text_similarity_score, current_score
                    )

                if attempt < self._compute_retries:
                    logging.info(
                        f"Retrying entire process in {self._compute_delay} seconds."
                    )
                    time.sleep(self._compute_delay)
                else:
                    error_message = (
                        "Exceeded maximum attempts to generate a GEST object that meets the similarity threshold. "
                        f"Expected minimum text similarity score is {self._improvement_text_similarity_threshold}. "
                        f"Minimum achieved text similarity score is {minimum_text_similarity_score}. "
                        f"Maximum achieved text similarity score is {maximum_text_similarity_score}."
                    )
                    logging.error(error_message)
                    raise GESTGenerationError(error_message) from err

        raise GESTGenerationError("Could not generate GEST object.")

    # Core Logic

    def _generate(
        self,
        llm_provider: BaseLLMProvider,
        content: str,
    ) -> GEST:
        """
        Generates a GEST object from the content, retrying on LLM or validation errors.
        """

        logging.info("Executing GEST generation step.")

        user_prompt = self._compute_generation_user_prompt(content=content)
        system_prompt = self._compute_generation_system_prompt(
            include_detailed_instructions=self._generation_include_detailed_instructions,
            include_few_shot_examples=self._generation_include_few_shot_examples,
        )

        def attempt_func(
            user_prompt: str,
            system_prompt: Optional[str],
        ) -> GEST:
            """
            Attempts to chat with the LLM provider and to return a valid GEST object.
            Since attempting means that it can fail, 'BaseLLMProviderResponseError' and 'ValidationError' should be tracked for error handling.
            """

            raw_response = llm_provider.chat(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                expected_result_type=LLMExpectedResultType.JSON,
            )

            return GEST.model_validate(raw_response)

        return self._execute_with_retries(
            attempt_func=attempt_func,
            initial_user_prompt=user_prompt,
            initial_system_prompt=system_prompt,
            retries=self._generation_retries,
            delay=self._generation_delay,
            operation_name=FlowOperationNameEnum.GENERATION,
        )

    def _improve(
        self,
        generated_gest: GEST,
        original_content: str,
    ) -> GEST:
        """
        Improves a GEST object, retrying on LLM/validation errors.
        Raises GESTContentSimilarityError if the result is not similar enough.
        """

        logging.info("Executing GEST improvement step.")

        try:
            no_sentences = (
                StringHelper().get_no_sentences(content=original_content)
                if self._improvement_include_exact_no_sentences
                else None
            )
            user_prompt = self._compute_improvement_user_prompt(
                content=generated_gest.model_dump_json(indent=2, exclude_none=True),
                no_sentences=no_sentences,
            )
        except GESTValueError as err:
            logging.warning(
                f"A GEST value error occurred while computing user prompt for GEST improvement. "
                f"Will use a default of 1 sentence. Error: '{err}'."
            )
            user_prompt = self._compute_improvement_user_prompt(
                content=original_content, no_sentences=1
            )

        system_prompt = self._compute_improvement_system_prompt(
            include_detailed_instructions=self._improvement_include_detailed_instructions,
            include_few_shot_examples=self._improvement_include_few_shot_examples,
        )

        def attempt_func_with_new_provider(
            user_prompt: str,
            system_prompt: Optional[str],
        ) -> GEST:
            """
            Attempts to improve the GEST object using a new LLM provider instance.
            This is necessary to ensure that we do not reuse the same provider instance
            that may have been affected by the previous generation attempt.
            """

            logging.debug(
                "Creating a new LLM provider instance for improvement attempt."
            )

            improve_llm_provider = LLMProviderFactory().create_provider(
                type=self._improvement_llm_provider_enum,
            )

            return self._attempt_improvement_chat(
                llm_provider=improve_llm_provider,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                original_content=original_content,
                generated_gest=generated_gest,
            )

        return self._execute_with_retries(
            attempt_func=attempt_func_with_new_provider,
            initial_user_prompt=user_prompt,
            initial_system_prompt=system_prompt,
            retries=self._improvement_retries,
            delay=self._improvement_delay,
            operation_name=FlowOperationNameEnum.IMPROVEMENT,
        )

    def _execute_with_retries(
        self,
        attempt_func: Callable[..., GEST],
        initial_user_prompt: str,
        initial_system_prompt: Optional[str],
        retries: int,
        delay: int,
        operation_name: FlowOperationNameEnum,
    ) -> GEST:
        """
        A generic helper to execute a function with a retry mechanism.

        It attempts to call `attempt_func`. On `BaseLLMProviderResponseError` or
        `ValidationError`, it computes a feedback prompt and retries.

        Args:
            attempt_func: The function to execute. Must accept 'user_prompt' and 'system_prompt'.
            initial_user_prompt: The first user prompt to use.
            initial_system_prompt: The first system prompt to use.
            retries: Maximum number of attempts.
            delay: Delay between retries in seconds.
            operation_name: A name for logging purposes (e.g., "generation").

        Returns:
            A valid GEST object.

        Raises:
            GESTGenerationError: If all retries fail.
            Any other exception raised by `attempt_func` that is not caught.
        """
        user_prompt, system_prompt = initial_user_prompt, initial_system_prompt

        for attempt in range(1, retries + 1):
            try:
                return attempt_func(
                    user_prompt=user_prompt, system_prompt=system_prompt
                )
            except (BaseLLMProviderResponseError, ValidationError) as err:
                logging.debug(
                    f"Attempt {attempt}/{retries} for '{operation_name}' failed with error: {err}"
                )
                user_prompt = self._compute_generation_feedback_prompt(
                    error=BaseLLMProviderResponseError(str(err))
                )
                system_prompt = None

                if attempt < retries:
                    logging.debug(f"Retrying in {delay} seconds.")
                    time.sleep(delay)

        error_message = (
            f"Exceeded maximum {retries} attempts for GEST {operation_name}."
        )
        logging.error(error_message)
        raise GESTGenerationError(error_message)

    def _attempt_improvement_chat(
        self,
        llm_provider: BaseLLMProvider,
        user_prompt: str,
        system_prompt: Optional[str],
        original_content: str,
        generated_gest: GEST,
    ) -> GEST:
        """
        Attempts to chat with the LLM provider and to return a valid GEST object.
        Since attempting means that it can fail, 'BaseLLMProviderResponseError' and 'GESTContentSimilarityError' should be tracked for error handling.
        """

        raw_narrative = llm_provider.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            expected_result_type=LLMExpectedResultType.STRING,
        )

        text_similarity_score = TextSimilarityEvaluator().compute_text_similarity(
            original=original_content,
            recomputed=raw_narrative,
        )

        if text_similarity_score >= self._improvement_text_similarity_threshold:
            logging.info(
                f"Text similarity score ({text_similarity_score:.4f}) is above the threshold "
                f"({self._improvement_text_similarity_threshold}). Returning GEST."
            )
            return generated_gest

        raise GESTContentSimilarityError(
            message=(
                f"Text similarity score ({text_similarity_score:.4f}) is below the threshold "
                f"({self._improvement_text_similarity_threshold}). Improvement is required."
            ),
            original_user_input=original_content,
            proposed_gest_content=generated_gest,
            recomputed_narrative=raw_narrative,
            current_text_similarity_score=text_similarity_score,
        )

    # Prompt Computation
    def _compute_generation_feedback_prompt(
        self,
        error: BaseLLMProviderResponseError,
        template_name: str = "generation/feedback_prompt.jinja",
    ) -> str:
        """Computes the generation feedback prompt using a Jinja template."""

        logging.info(f"Computing generation feedback prompt for error: {error}")

        feedback_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        return feedback_prompt_template.render(
            error_message=str(error),
        )

    def _compute_generation_improve_representation_prompt(
        self,
        error: GESTContentSimilarityError,
        template_name: str = "generation/improve_representation.jinja",
    ) -> str:
        """Computes the generation improve representation prompt using a Jinja template."""

        logging.info(
            f"Computing generation improve representation prompt for error: {error}"
        )

        improve_representation_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        return improve_representation_prompt_template.render(
            text_similarity_threshold=self._improvement_text_similarity_threshold,
            original_user_input=error.original_user_input,
            proposed_gest_content=error.proposed_gest_content,
            recomputed_narrative=error.recomputed_narrative,
            current_text_similarity_score=error.current_text_similarity_score,
        )

    def _compute_generation_system_prompt(
        self,
        template_name: str = "generation/system_prompt.jinja",
        include_detailed_instructions: bool = True,
        include_few_shot_examples: bool = True,
    ) -> str:
        """Computes the system prompt using a Jinja template."""

        system_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        return system_prompt_template.render(
            output_schema_json_str=json.dumps(
                GEST.model_json_schema(),
                indent=2,
            ),
            include_detailed_instructions=include_detailed_instructions,
            include_few_shot_examples=include_few_shot_examples,
        )

    def _compute_generation_user_prompt(
        self,
        content: str,
        template_name: str = "generation/user_prompt.jinja",
    ) -> str:
        """Computes the user prompt using a Jinja template."""

        logging.info(f"Computing user prompt for content: {content}")

        user_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        return user_prompt_template.render(
            content=content,
        )

    def _compute_improvement_system_prompt(
        self,
        template_name: str = "improvement/system_prompt.jinja",
        include_detailed_instructions: bool = True,
        include_few_shot_examples: bool = True,
    ) -> str:
        """Computes the system prompt for GEST content improvement using a Jinja template."""

        system_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        return system_prompt_template.render(
            input_schema_json_str=json.dumps(
                GEST.model_json_schema(),
                indent=2,
            ),
            include_exact_no_sentences=self._improvement_include_exact_no_sentences,
            include_detailed_instructions=include_detailed_instructions,
            include_few_shot_examples=include_few_shot_examples,
        )

    def _compute_improvement_user_prompt(
        self,
        content: str,
        no_sentences: Optional[int] = None,
        template_name: str = "improvement/user_prompt.jinja",
    ) -> str:
        """Computes the user prompt using a Jinja template."""

        logging.info(
            f"Computing user prompt for content: {content}. "
            f"Including exact number of sentences set value is '{self._improvement_include_exact_no_sentences}'. "
            f"Number of sentences value is '{no_sentences}'."
        )

        user_prompt_template = self._jinja_helper.get_template(
            template_name=template_name,
        )

        if self._improvement_include_exact_no_sentences:
            if no_sentences is None:
                message = "Number of sentences value must be provided when 'include_exact_no_sentences' is set on True."
                logging.exception(message)
                raise GESTValueError(message)

            return user_prompt_template.render(
                content=content,
                include_exact_no_sentences=True,
                no_sentences=no_sentences,
            )

        return user_prompt_template.render(
            content=content,
            include_exact_no_sentences=False,
        )

    # Config helpers
    @staticmethod
    def _get_configs() -> AttrDict:
        """Get the default GEST configs from the configuration."""

        gest_configs = ConfigLoader().get("gest.engine")

        if isinstance(gest_configs, AttrDict):
            return gest_configs

        raise ConfigurationError("Invalid configuration format for GEST engine.")

    @staticmethod
    def _get_generation_configs() -> AttrDict:
        """Get the default GEST engine generation configs from the configuration."""

        gest_generation_configs = ConfigLoader().get("gest.engine.generation")

        if isinstance(gest_generation_configs, AttrDict):
            return gest_generation_configs

        raise ConfigurationError(
            "Invalid configuration format for GEST engine generation."
        )

    @staticmethod
    def _get_improvement_configs() -> AttrDict:
        """Get the default GEST engine improvement configs from the configuration."""

        gest_improvement_configs = ConfigLoader().get("gest.engine.improvement")

        if isinstance(gest_improvement_configs, AttrDict):
            return gest_improvement_configs

        raise ConfigurationError(
            "Invalid configuration format for GEST engine improvement."
        )

    def _assign_configs(self) -> None:
        """Assign GEST engine configs from the configuration."""
        configs = self._get_configs()
        generation_configs = self._get_generation_configs()
        improvement_configs = self._get_improvement_configs()

        # Engine configs
        self._compute_retries = self._assign_retries_config(configs)
        self._compute_delay = self._assign_delay_config(configs)

        # Generation configs
        self._generation_llm_provider_enum = self._assign_llm_provider_config(
            generation_configs
        )
        self._generation_retries = self._assign_retries_config(generation_configs)
        self._generation_delay = self._assign_delay_config(generation_configs)
        self._generation_include_detailed_instructions = (
            self._assign_include_detailed_instructions_config(generation_configs)
        )
        self._generation_include_few_shot_examples = (
            self._assign_include_few_shot_examples_config(generation_configs)
        )

        # Improvement configs
        self._improvement_llm_provider_enum = self._assign_llm_provider_config(
            improvement_configs
        )
        self._improvement_skip_step = self._assign_skip_step_config(improvement_configs)
        self._improvement_retries = self._assign_retries_config(improvement_configs)
        self._improvement_delay = self._assign_delay_config(improvement_configs)
        self._improvement_text_similarity_threshold = (
            self._assign_text_similarity_threshold_config(improvement_configs)
        )
        self._improvement_include_exact_no_sentences = (
            self._assign_include_exact_no_sentences_config(improvement_configs)
        )
        self._improvement_include_detailed_instructions = (
            self._assign_include_detailed_instructions_config(improvement_configs)
        )
        self._improvement_include_few_shot_examples = (
            self._assign_include_few_shot_examples_config(improvement_configs)
        )

    def _assign_llm_provider_config(self, configs: AttrDict) -> LLMProviderEnum:
        """Assign the llm provider configuration for GEST engine."""

        llm_provider = configs.llm_provider

        if not isinstance(llm_provider, str):
            raise ConfigurationError(
                "Invalid LLM Provider for GEST engine. LLM Provider value must be a string."
            )

        try:
            llm_provider = LLMProviderEnum(llm_provider)
        except KeyError:
            raise ConfigurationError(
                f"Invalid LLM Provider '{llm_provider}' for GEST engine."
                f"Available providers are {', '.join([provider.value for provider in LLMProviderEnum])}."
            )

        return llm_provider

    def _assign_skip_step_config(self, configs: AttrDict) -> bool:
        """Assign the skip step configuration for GEST engine."""

        skip_step = configs.skip_improvement_step

        if not isinstance(skip_step, bool):
            raise ConfigurationError(
                "Invalid 'skip_step' for GEST engine. Provided value must be a boolean."
            )

        return skip_step

    def _assign_retries_config(self, configs: AttrDict) -> int:
        """Assign the retries configuration for GEST engine."""

        retries = configs.retries

        if not isinstance(retries, int):
            raise ConfigurationError(
                "Invalid retries for GEST engine. Provided value must be an integer."
            )

        return retries

    def _assign_delay_config(self, configs: AttrDict) -> int:
        """Assign the delay configuration for GEST engine."""

        delay = configs.delay

        if not isinstance(delay, int):
            raise ConfigurationError(
                "Invalid delay for GEST engine. Provided value must be an integer."
            )

        return delay

    def _assign_include_detailed_instructions_config(self, configs: AttrDict) -> bool:
        """Assign the include detailed instructions configuration for GEST engine."""

        include_detailed_instructions = configs.include_detailed_instructions

        if not isinstance(include_detailed_instructions, bool):
            raise ConfigurationError(
                "Invalid 'include_detailed_instructions' for GEST engine. Provided value must be a boolean."
            )

        return include_detailed_instructions

    def _assign_include_few_shot_examples_config(self, configs: AttrDict) -> bool:
        """Assign the include few shot examples configuration for GEST engine."""

        include_few_shot_examples = configs.include_few_shot_examples

        if not isinstance(include_few_shot_examples, bool):
            raise ConfigurationError(
                "Invalid 'include_few_shot_examples' for GEST engine. Provided value must be a boolean."
            )

        return include_few_shot_examples

    def _assign_text_similarity_threshold_config(self, configs: AttrDict) -> float:
        """Assign the text similarity threshold configuration for GEST engine."""

        raw = configs.text_similarity_threshold

        if not isinstance(raw, float):
            raise ConfigurationError(
                "Invalid text similarity threshold for GEST engine. "
                f"Provided value ('{raw}') must be a float."
            )

        try:
            text_similarity_threshold = cast(float, raw)
        except Exception as e:
            raise ConfigurationError(
                "An exception occurred while casting text similarity threshold to float. "
                f"Error: {e}"
            )

        if not (0 <= text_similarity_threshold <= 1):
            raise ConfigurationError(
                "Invalid text similarity threshold for GEST engine. "
                "Provided value must be between 0 and 1."
            )

        return text_similarity_threshold

    def _assign_include_exact_no_sentences_config(self, configs: AttrDict) -> bool:
        """Assign the include exact no sentences configuration for GEST engine."""

        include_exact_no_sentences = configs.include_exact_no_sentences

        if not isinstance(include_exact_no_sentences, bool):
            raise ConfigurationError(
                "Invalid retries for GEST engine. Provided value must be a boolean."
            )

        return include_exact_no_sentences
