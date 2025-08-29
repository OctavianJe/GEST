import logging
import os
from pathlib import Path
from typing import List

from gest.common.helpers.config_loader import ConfigurationError
from gest.common.helpers.gmail_helper import GmailHelper
from gest.common.helpers.time_helper import TimeHelper
from gest.dataset.source.activity_net_dataset import ActivityNetCaptionsDataset
from gest.dataset.source.base_source_dataset import SourceDataset
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
from gest.service.other.llm.provider.exception.provider_request_exceptions import (
    BaseLLMProviderRequestException,
)


def _get_blacklist_dataset() -> GestBlacklistDataset:
    """Defines the `GestBlacklistDataset` used to store blacklisted rows."""
    return GestBlacklistDataset(csv_path=Path("/workspaces/GEST/data/blacklist.csv"))


def _get_destination_dataset() -> GestDataset:
    """Defines the `GestDataset` used to store GEST."""
    return GestDataset(csv_path=Path("/workspaces/GEST/data/gest.csv"))


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


if __name__ == "__main__":
    try:
        datasets: List[SourceDataset] = _get_source_datasets()

        gest: GestDataset = _get_destination_dataset()
        blacklist: GestBlacklistDataset = _get_blacklist_dataset()

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
