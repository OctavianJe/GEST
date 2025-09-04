import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List

from gest.common.enums.gest_task_enum import GestTaskEnum
from gest.common.helpers.attr_dict import AttrDict
from gest.common.helpers.config_loader import ConfigLoader, ConfigurationError
from gest.common.helpers.gmail_helper import GmailHelper
from gest.common.helpers.time_helper import TimeHelper
from gest.dataset.source.activity_net_dataset import ActivityNetCaptionsDataset
from gest.dataset.source.base_source_dataset import SourceDataset
from gest.dataset.source.source_dataset_enum import SourceDatasetEnum
from gest.dataset.target.gest_dataset import (
    GestBlacklistDataset,
    GestBlacklistRow,
    GestDataset,
    GestRow,
)
from gest.service.generation.exception.gest_exceptions import (
    GESTContentSimilarityError,
    GESTGenerationError,
)
from gest.service.generation.gest_engine import GESTEngine
from gest.service.other.llm.provider.base_llm_provider import LLMProviderEnum
from gest.service.other.llm.provider.exception.provider_request_exceptions import (
    BaseLLMProviderRequestException,
)


def _get_blacklist_dataset(path: str) -> GestBlacklistDataset:
    """Defines the `GestBlacklistDataset` used to store blacklisted rows."""
    return GestBlacklistDataset(csv_path=Path(f"{path}/blacklist.csv"))


def _get_destination_dataset(path: str) -> GestDataset:
    """Defines the `GestDataset` used to store GEST."""
    return GestDataset(csv_path=Path(f"{path}/gest.csv"))


def _get_evaluation_destination_dataset(name: str, path: str) -> GestDataset:
    """GestDataset for evaluation outputs."""
    return GestDataset(csv_path=Path(f"{path}/gest_eval_{name}.csv"))


def _get_evaluation_blacklist_dataset(name: str, path: str) -> GestBlacklistDataset:
    """BlacklistDataset for evaluation runs."""
    return GestBlacklistDataset(csv_path=Path(f"{path}/blacklist_eval_{name}.csv"))


def _get_manual_annotation_dataset(path: str) -> GestDataset:
    """Defines the `GestDataset` that stores manual annotations."""
    return GestDataset(csv_path=Path(f"{path}/gest_manual.csv"))


def _get_generation_flow_model_config() -> str:
    """Get the generation flow model name from configuration of GEST."""
    llm_provider = ConfigLoader().get("gest.engine.generation.llm_provider")

    if not isinstance(llm_provider, str):
        raise ConfigurationError(
            "Invalid LLM provider for GEST engine. LLM provider value must be a string."
        )

    try:
        provider_enum = LLMProviderEnum(llm_provider)
    except KeyError:
        raise ConfigurationError(
            f"Invalid LLM Provider '{llm_provider}' for GEST engine."
            f"Available providers are {', '.join([provider.value for provider in LLMProviderEnum])}."
        )

    llm_configs = ConfigLoader().get("llm")

    if not isinstance(llm_configs, AttrDict):
        raise ConfigurationError("Invalid LLM configuration for GEST engine.")

    provider_cfg = llm_configs.get(provider_enum.value)
    if provider_cfg is None:
        raise ConfigurationError(
            f"Missing LLM provider configuration at key 'llm.{provider_enum.value}'."
        )

    generation_flow_model_name = provider_cfg.get("model")
    if not isinstance(generation_flow_model_name, str):
        raise ConfigurationError("Invalid generation flow model name for GEST engine.")

    return generation_flow_model_name.partition(":")[0]


def _get_source_datasets() -> List[SourceDataset]:
    """Defines all `SourceDataset` used to create GEST."""
    datasets: List[SourceDataset] = []

    anc = ActivityNetCaptionsDataset(
        Path("/workspaces/GEST/miscellaneous/datasets/ActivityNet Captions/train.json"),
        Path("/workspaces/GEST/miscellaneous/datasets/ActivityNet Captions/val_1.json"),
        Path("/workspaces/GEST/miscellaneous/datasets/ActivityNet Captions/val_2.json"),
    )
    datasets.append(anc)

    return datasets


def _get_data_path() -> str:
    data_path = ConfigLoader().get("gest.data_path")

    if not isinstance(data_path, str):
        raise ConfigurationError(
            "Invalid data path for GEST engine. Data path value must be a string."
        )

    return data_path


def _get_manual_path() -> str:
    manual_path = ConfigLoader().get("gest.manual_path")

    if not isinstance(manual_path, str):
        raise ConfigurationError(
            "Invalid manual path for GEST engine. Manual path value must be a string."
        )

    return manual_path


def _get_blacklist_path() -> str:
    blacklist_path = ConfigLoader().get("gest.blacklist_path")

    if not isinstance(blacklist_path, str):
        raise ConfigurationError(
            "Invalid blacklist path for GEST engine. Blacklist path value must be a string."
        )

    return blacklist_path


def _get_task_config() -> GestTaskEnum:
    """Get the task from configuration of GEST."""
    task = ConfigLoader().get("gest.task")

    if not isinstance(task, str):
        raise ConfigurationError(
            "Invalid task for GEST engine. Task value must be a string."
        )

    try:
        return GestTaskEnum(task)
    except ValueError:
        raise ConfigurationError(
            f"Invalid task '{task}' for GEST engine."
            f"Available tasks are {', '.join([task.value for task in GestTaskEnum])}."
        )


def _intersect_sources_with_manual(
    sources: List[SourceDataset],
    manual: GestDataset,
) -> List[SourceDataset]:
    """
    Filter each SourceDataset's .rows in place to keep only samples whose
    (dataset, id) appear in the manual annotations.
    """
    ds_col = GestRow.fields().dataset
    id_col = GestRow.fields().id

    allowed: dict[SourceDatasetEnum, set[str]] = defaultdict(set)
    for _, r in manual.df[[ds_col, id_col]].iterrows():
        ds_val = r[ds_col]
        ds_enum = (
            ds_val
            if isinstance(ds_val, SourceDatasetEnum)
            else SourceDatasetEnum(ds_val)
        )
        allowed[ds_enum].add(str(r[id_col]))

    filtered_sources: List[SourceDataset] = []
    for src in sources:
        ids_for_dataset = allowed.get(src.name, set())
        if not ids_for_dataset:
            continue

        src.rows = [row for row in src.rows if row.id() in ids_for_dataset]

        if src.rows:
            filtered_sources.append(src)

    return filtered_sources


def main():
    try:
        task = _get_task_config()
        data_path = _get_data_path()
        manual_path = _get_manual_path()
        blacklist_path = _get_blacklist_path()

        if task == GestTaskEnum.GENERATION:
            datasets: List[SourceDataset] = _get_source_datasets()

            gest: GestDataset = _get_destination_dataset(path=data_path)
            blacklist: GestBlacklistDataset = _get_blacklist_dataset(
                path=blacklist_path
            )

        elif task == GestTaskEnum.EVALUATION:
            generation_flow_model_name = _get_generation_flow_model_config()

            datasets: List[SourceDataset] = _intersect_sources_with_manual(
                sources=_get_source_datasets(),
                manual=_get_manual_annotation_dataset(path=manual_path),
            )

            gest: GestDataset = _get_evaluation_destination_dataset(
                name=generation_flow_model_name,
                path=data_path,
            )
            blacklist: GestBlacklistDataset = _get_evaluation_blacklist_dataset(
                name=generation_flow_model_name,
                path=blacklist_path,
            )

        else:
            raise ConfigurationError(
                f"Unsupported GEST '{task}' configuration task. "
                f"Available tasks are {', '.join([task.value for task in GestTaskEnum])}."
            )

        for dataset in datasets:
            for row in dataset.rows:
                id = row.id()

                exists = gest.find_row_index(source_dataset=dataset.name, id=id)
                blacklisted = blacklist.is_blacklisted(dataset=dataset.name, id=id)

                if exists is None and not blacklisted:
                    try:
                        text = row.text()

                        try:
                            gest_representation = GESTEngine().compute(content=text)
                        except GESTGenerationError as e:
                            logging.warning(
                                f"GEST object could not be created for dataset '{dataset.name}' with id '{id}'. "
                                f"Text content used: '{text}'. "
                                f"Encountered error: '{e}'."
                            )
                            blacklist.append_row_to_csv(
                                GestBlacklistRow(
                                    dataset=dataset.name,
                                    id=id,
                                    reason=str(e),
                                )
                            )
                            continue
                        except GESTContentSimilarityError as e:
                            logging.warning(
                                f"GEST object content could not be above the minimum defined 'text_similarity_threshold' for dataset '{dataset.name}' with id '{id}'. "
                                f"Text content used: '{text}'. "
                                f"Encountered error: '{e}'."
                            )
                            blacklist.append_row_to_csv(
                                GestBlacklistRow(
                                    dataset=dataset.name,
                                    id=id,
                                    reason=str(e),
                                )
                            )
                            continue

                        gest.append_row_to_csv(
                            GestRow(
                                dataset=dataset.name,
                                id=id,
                                text=text,
                                gest=gest_representation,
                            )
                        )
                    except ConfigurationError as e:
                        logging.exception(
                            f"Configuration file is not set up correctly while processing dataset '{dataset.name}' with id '{id}'. "
                            "Will stop processing any further rows. "
                            f"Encountered error: {e}"
                        )
                        raise Exception(e) from e
                    except BaseLLMProviderRequestException as e:
                        logging.exception(
                            f"All available backup API Keys exhausted while processing dataset '{dataset.name}' with id '{id}'. "
                            "Will stop processing any further rows. "
                            f"Exception is '{e}'."
                        )
                        raise Exception(e) from e
                    except Exception as e:
                        logging.exception(
                            f"An unexpected exception occurred while processing dataset '{dataset.name}' with id '{id}'. "
                            "Will continue with the next row. "
                            f"Exception is '{e}'."
                        )
                        blacklist.append_row_to_csv(
                            GestBlacklistRow(
                                dataset=dataset.name,
                                id=id,
                                reason=str(e),
                            )
                        )
                        continue

            logging.info(f"GEST generation completed for dataset '{dataset.name}'.")

            # Will send an email alert only if enabled from configuration.
            GmailHelper().send_email(
                subject=f"[GEST] Generation for '{dataset.name}' dataset completed",
                body=(
                    f"GEST engine successfully completed the generation for '{dataset.name}' dataset at '{TimeHelper().get_now_local()}'."
                    "Will now continue with the next dataset. "
                ),
            )

        logging.info("GEST generation completed for all datasets.")

        # Will send an email alert only if enabled from configuration.
        GmailHelper().send_email(
            subject="[GEST] Generation for all queued dataset is completed",
            body=(
                f"GEST engine successfully completed the generation of all dataset at '{TimeHelper().get_now_local()}'."
            ),
        )

    except Exception as e:
        logging.exception(
            f"An exception occurred. Will gracefully exit. Exception is '{e}'."
        )
        # Will send an email alert only if enabled from configuration.
        GmailHelper().send_email(
            subject="[GEST] Generation process stopped due to an exception",
            body=(
                f"GEST engine stopped at '{TimeHelper().get_now_local()}'."
                f"Exception encountered: '{e}'."
            ),
        )
        os._exit(1)


if __name__ == "__main__":
    main()
