from datetime import datetime
import pathlib
import yaml
from typing import Callable, Dict, Optional, Sequence, Any
from threading import Lock

from pyrit.models import SeedPromptDataset, SeedPrompt
from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import (
    harmbench_dataset,
    darkbench_dataset,
    forbidden_questions_dataset,
    librAI_do_not_answer_dataset,
    red_team_social_bias_dataset
)

from logging_handler import logger, run_only_if_log_level_debug

_default_custom_dataset_meta = {
    "dataset_name": "pyrit_test_framework_dataset",
    "harm_categories": "child_safety",
    "source": "pyrit_test_framework",
    "authors": ["Ferrero Toys Red Teaming"],
    "groups": "Ferrero Toys Red Teaming",
    "description": "Potentially harmful for children"
}

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

def enrich_dataset_info_from_prompts(dataset: SeedPromptDataset) -> SeedPromptDataset:
    """
    Enrich dataset info (if missing) using info stored in seed prompts or filename.
    """
    if dataset.prompts and len(dataset.prompts) > 0:
        template_prompt = dataset.prompts[0]
        dataset.authors = dataset.authors or template_prompt.authors
        dataset.groups = dataset.groups or template_prompt.groups
        dataset.source = dataset.source or template_prompt.groups
        dataset.harm_categories = dataset.harm_categories or template_prompt.harm_categories

        dataset.dataset_name = dataset.dataset_name or template_prompt.dataset_name
        dataset.description = dataset.description or template_prompt.description

    return dataset

def _build_seed_prompt_dataset_from_custom(
    *,
    prompts: Sequence[str],
    dataset_name: str = _default_custom_dataset_meta['dataset_name'],
    description: str = _default_custom_dataset_meta['description'],
    harm_categories: Sequence[str] = _default_custom_dataset_meta['harm_categories'],
    authors: Sequence[str] = _default_custom_dataset_meta['authors'],
    groups: Sequence[str] = _default_custom_dataset_meta['groups'],
    source: str = _default_custom_dataset_meta['source'],
    date_added: datetime = datetime.now()
) -> SeedPromptDataset:
    prompts = [SeedPrompt(value=p, data_type="text") for p in prompts]
    return SeedPromptDataset(
        dataset_name=dataset_name,
        description=description,
        harm_categories=harm_categories,
        authors=authors,
        groups=groups,
        source=source,
        date_added=date_added,
        prompts=prompts
    )

def load_txt_prompts_file(txt_path: pathlib.Path) -> SeedPromptDataset:
    prompts = []
    with txt_path.open("r", encoding="utf-8") as f:
        prompts.extend([line.strip() for line in f if line.strip()])

    return _build_seed_prompt_dataset_from_custom(
        prompts=prompts,
        dataset_name=f"{txt_path.stem}_txt",
        description=f"{_default_custom_dataset_meta['description']}. Custom dataset loaded from {txt_path.name}"
    )

def load_yaml_prompts_file(yaml_path: pathlib.Path) -> SeedPromptDataset:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # If the YAML is just a list, treat it as a list of prompts
    if isinstance(data, list):
        prompts = [SeedPrompt(value=p, data_type="text") for p in data]
        return _build_seed_prompt_dataset_from_custom(
            prompts=prompts,
            dataset_name=f"{yaml_path.stem}_yaml",
            description=f"{_default_custom_dataset_meta['description']}. Custom dataset loaded from {yaml_path.name}"
        )

    # If the YAML is a dict, try to use the fields
    elif isinstance(data, dict):
        # Filter only keys in params that exist in the function signature
        from inspect import signature
        sig = signature(_build_seed_prompt_dataset_from_custom)
        valid_params = {k: v for k, v in data.items() if k in sig.parameters}
        valid_params['dataset_name'] = data.get("dataset_name", f"{yaml_path.stem}_yaml")
        valid_params['description'] = data.get("description", f"{_default_custom_dataset_meta['description']}. Custom dataset loaded from {yaml_path.name}")

        # add to args if loaded a filled value from yaml file, otherwise get builder defaults
        return _build_seed_prompt_dataset_from_custom(**valid_params)

    else:
        raise ValueError('Unrecognized type loaded from custom prompts yaml (not dict nor list)')

class LazyLoader:
    def __init__(self, loaders_with_flags: Dict[str, tuple[Callable[[], SeedPromptDataset], bool]]):
        self._should_cache = {k: v[1] for k, v in loaders_with_flags.items()}
        self._loaders = {k: v[0] for k, v in loaders_with_flags.items()}
        self._cache = {}
        self._locks = {key: Lock() for key in self._loaders}

    def get(self, key):
        if key in self._cache:
            return self._cache[key]
        if key not in self._loaders:
            raise KeyError(f"Dataset '{key}' not found")
        # Use a lock per key to prevent race conditions
        with self._locks[key]:
            if key in self._cache:
                return self._cache[key]
            dataset = self._loaders[key]()
            dataset = enrich_dataset_info_from_prompts(dataset)
            if self._should_cache.get(key, True):
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

def discover_custom_datasets(custom_dir: pathlib.Path) -> Dict[str, tuple[Callable[[], SeedPromptDataset], bool]]:
    """
    Discover all .yaml and .txt files in the custom_dir and return a dict mapping
    dataset_name -> (loader function, should_cache)
    """
    loaders = {}
    for file in custom_dir.glob("*.yaml"):
        dataset_name = f"{file.stem}_yaml"
        loaders[dataset_name] = (lambda f=file: load_yaml_prompts_file(f), True)
    for file in custom_dir.glob("*.txt"):
        dataset_name = f"{file.stem}_txt"
        loaders[dataset_name] = (lambda f=file: load_txt_prompts_file(f), True)
    return loaders

# External datasets (from pyrit)
# the dataset loaded by pyrit don't need a custom cache as they manage caching themselves. But the lazy loader might be useful for custom datasets.
# dict dataset-name : tuple(should_cache, fetcher of SeedPromptDataset) (lazy loading)
_external_loaders = {
    "illegal": (lambda: SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH)/"seed_prompts"/"illegal.prompt"), True),
    "harmbench": (lambda: harmbench_dataset.fetch_harmbench_dataset(), False),
    "darkbench": (lambda: darkbench_dataset.fetch_darkbench_dataset(), False),
    "forbidden": (lambda: forbidden_questions_dataset.fetch_forbidden_questions_dataset(), False),
    "librai": (lambda: librAI_do_not_answer_dataset.fetch_librAI_do_not_answer_dataset(), False),
    "bias": (lambda: red_team_social_bias_dataset.fetch_red_team_social_bias_dataset(), False),
}

_lazy_loader = None

def set_dataset_loader(loader: LazyLoader):
    global _lazy_loader
    _lazy_loader = loader

def initialize_dataset_loader(config: dict) -> LazyLoader:
    # Custom datasets
    custom_dir = pathlib.Path(config["datasets"]["custom"]["dir"])
    custom_loaders = discover_custom_datasets(custom_dir)
    # Merge
    all_loaders = {**_external_loaders, **custom_loaders}
    set_dataset_loader(LazyLoader(all_loaders))
    return custom_loaders.keys()

def list_available_datasets() -> Sequence[str]:
    if not _lazy_loader:
        raise RuntimeError('Dataset Loader was not initialized yet!')
    return list(_lazy_loader._loaders.keys())

def load_dataset(*, dataset_name: str) -> SeedPromptDataset:
    if not _lazy_loader:
        raise RuntimeError('Dataset Loader was not initialized yet!')
    return _lazy_loader.get(dataset_name)