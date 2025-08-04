from pyrit.models import SeedPromptDataset
from pyrit.common.path import DATASETS_PATH

from pyrit.datasets import (
    harmbench_dataset,
    darkbench_dataset,
    forbidden_questions_dataset,
    librAI_do_not_answer_dataset,
    red_team_social_bias_dataset
)

import pathlib
from threading import Lock

# TODO improve with singleton pattern, enums for datasets etc.
# TODO IMPORTANT: this might be useless as pyrit fetchers already manage caching with local db
# we might add a "cachable" flag just to handle our custom datasets, and leave pyrit manage its own caching mechanism
class LazyLoader:
    def __init__(self, loaders):
        """
        loaders: dict[str, callable] that return the heavy dataset on call.
        """
        self._loaders = loaders
        self._cache = {}
        self._locks = {key: Lock() for key in loaders}

    def get(self, key):
        # Return cached if available
        if key in self._cache:
            return self._cache[key]

        if key not in self._loaders:
            raise KeyError(f"Dataset '{key}' not found")

        # Use a lock per key to prevent race conditions on loading
        with self._locks[key]:
            # Double-check after acquiring lock
            if key in self._cache:
                return self._cache[key]
            # Load dataset heavy operation
            dataset = self._loaders[key]()
            self._cache[key] = dataset
            return dataset

    def preload(self, key):
        """
        Force loading of dataset and caching without waiting for user access.
        """
        return self.get(key)

    def clear_cache(self, key=None):
        """
        Clear cached datasets. If key is None, clears all.
        """
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def is_loaded(self, key):
        """
        Check if a dataset is already loaded and cached.
        """
        return key in self._cache

# dict dataset-name : closure of SeedPromptDataset (lazy loading)
_datasets = {
    "illegal": lambda: SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH)/"seed_prompts"/"illegal.prompt"),
    "harmbench": lambda: harmbench_dataset.fetch_harmbench_dataset(),
    "darkbench": lambda: darkbench_dataset.fetch_darkbench_dataset(),
    "forbidden": lambda: forbidden_questions_dataset.fetch_forbidden_questions_dataset(),
    "librai": lambda: librAI_do_not_answer_dataset.fetch_librAI_do_not_answer_dataset(),
    "bias": lambda: red_team_social_bias_dataset.fetch_red_team_social_bias_dataset(),
}

lazy_dataset_loader = LazyLoader(_datasets)

def load_dataset(*, dataset_name: str) -> SeedPromptDataset:
    return lazy_dataset_loader.get(dataset_name)
