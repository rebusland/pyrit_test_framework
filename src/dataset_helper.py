from pyrit.models import SeedPromptDataset
from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import (
    harmbench_dataset,
    darkbench_dataset,
    forbidden_questions_dataset,
    librAI_do_not_answer_dataset,
    red_team_social_bias_dataset
)

from logging_handler import logger
from utils import run_only_if_log_level_debug

import pathlib
from threading import Lock

@run_only_if_log_level_debug()
def peek_dataset_info(dataset: SeedPromptDataset):
    logger.debug(f"\n\n**** Peek some info on dataset {dataset.dataset_name} ****\n")
    logger.debug(f"Dataset description: {dataset.description}")
    logger.debug(f"Dataset name: {dataset.dataset_name}")
    logger.debug(f"Dataset data type: {dataset.data_type}")
    logger.debug(f"Dataset authors:' + {dataset.authors}")
    logger.debug(f"Dataset harm categories: {dataset.harm_categories}")
    logger.debug(f"Dataset total number of prompts: {len(dataset.prompts)}")
    logger.debug(f"Dataset Seed prompts (first 3 examples):\n {dataset.get_values()[:3] if len(dataset.get_values()) >= 3 else dataset.get_values()}\n")
    logger.debug(f"Dataset prompt groups: {dataset.groups}")

def enrich_dataset_info(dataset: SeedPromptDataset) -> None:
    '''
    Enrich dataset info (if missing) using info stored in seed prompts (useful for datasets retrieved using Pyrit APIs)
    '''
    # TODO
    return dataset

class LazyLoader:
    def __init__(self, loaders_with_flags: dict[str, tuple[callable, bool]]):
        """
        loaders_with_flags: dict mapping dataset key to (loader function, should_cache bool)
        Only puts in cache what's specified
        """
        self._should_cache = {k: v[0] for k, v in loaders_with_flags.items()}
        self._loaders = {k: v[1] for k, v in loaders_with_flags.items()}
        self._cache = {}
        self._locks = {key: Lock() for key in self._loaders}

    def get(self, key):
        # Return cached if available
        if key in self._cache:
            return self._cache[key]

        if key not in self._loaders:
            raise KeyError(f"Dataset '{key}' not found")

        # Use a lock per key to prevent race conditions
        with self._locks[key]:
            if key in self._cache:
                return self._cache[key]
            # enrich dataset info if missing
            dataset = enrich_dataset_info(self._loaders[key]())

            if self._should_cache.get(key, True):  # default to True
                self._cache[key] = dataset
            return dataset

    def preload(self, key):
        return self.get(key)

    def clear_cache(self, key=None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def is_loaded(self, key):
        return key in self._cache

# the dataset loaded by pyrit don't need a custom cache as they manage caching themselves. But the lazy loader might be useful for custom datasets.
# dict dataset-name : tuple(should_cache, fetcher of SeedPromptDataset) (lazy loading)
_datasets = {
    "illegal": (True, lambda: SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH)/"seed_prompts"/"illegal.prompt")),
    "harmbench": (False, lambda: harmbench_dataset.fetch_harmbench_dataset()),
    "darkbench": (False, lambda: darkbench_dataset.fetch_darkbench_dataset()),
    "forbidden": (False, lambda: forbidden_questions_dataset.fetch_forbidden_questions_dataset()),
    "librai": (False, lambda: librAI_do_not_answer_dataset.fetch_librAI_do_not_answer_dataset()),
    "bias": (False, lambda: red_team_social_bias_dataset.fetch_red_team_social_bias_dataset()),
}

lazy_dataset_loader = LazyLoader(_datasets)

def load_dataset(*, dataset_name: str) -> SeedPromptDataset:
    return lazy_dataset_loader.get(dataset_name)
