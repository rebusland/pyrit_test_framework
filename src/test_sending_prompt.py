from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import SeedPromptDataset
from pyrit.models import PromptRequestResponse
# from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_target import OpenAIChatTarget

from pyrit.datasets import (
    dataset_helper,
    harmbench_dataset,
    darkbench_dataset,
    forbidden_questions_dataset,
    librAI_do_not_answer_dataset,
    red_team_social_bias_dataset
)

import config_loader
import scorer_factory
import orchestrator_factory

import logging
import asyncio
import pathlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

config_loader.load_dotenv_with_check()
config_loader.load_openai_configs()

def peek_prompts_and_other_info(seed_prompts_dataset, memory):
    logger.debug('Datasets path:', pathlib.Path(DATASETS_PATH))

    logger.debug('Dataset description:' + seed_prompts_dataset.description)
    logger.debug('Dataset name:' + seed_prompts_dataset.dataset_name)
    logger.debug('Dataset data type:' + seed_prompts_dataset.data_type)
    logger.debug('Dataset authors:' + str(seed_prompts_dataset.authors))
    logger.debug('Dataset harm categories:' + str(seed_prompts_dataset.harm_categories))
    logger.debug('Seed prompts:' + str(seed_prompts_dataset.get_values()))

    logger.debug('Prompt groups:' + str(seed_prompts_dataset.groups))
    logger.debug('Prompt seed groups loaded in memory:' + str(memory.get_seed_prompt_groups()))
    for g in memory.get_seed_prompt_groups():
        logger.debug(str(g.prompts))

def get_report(results):
    pass

def load_seed_prompts(dataset):
    return datasets['dataset']

# dict key : SeedPromptDataset
datasets = {
    "illegal": SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal.prompt"),
    "harmbench": harmbench_dataset.fetch_harmbench_dataset(),
    "darkbench": darkbench_dataset.fetch_darkbench_dataset(),
    "forbidden": forbidden_questions_dataset.fetch_forbidden_questions_dataset(),
    "librai": librAI_do_not_answer_dataset.fetch_librAI_do_not_answer_dataset(),
    "bias": red_team_social_bias_dataset.fetch_red_team_social_bias_dataset(),
}

async def run_test_sending_prompts():
    # Initialize memory. We use duckDB
    initialize_pyrit(memory_db_type=IN_MEMORY)
    memory = CentralMemory.get_memory_instance()

    # Load dataset and prepare prompts
    seed_prompts_dataset = load_seed_prompts()
    seed_prompts = seed_prompts_dataset.prompts
    await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="pyrit_test_framework")
    # Configure the labels you want to send, these should be unique to this test to make it easier to retrieve
    memory_labels = {"op_name": "new_op", "user_name": "PyRIT_test_framework", "test_name": __file__}

    peek_prompts_and_other_info(seed_prompts_dataset=seed_prompts_dataset, memory=memory)

    prompts=[p.value for p in seed_prompts]

    prepended_prompt=config_loader.load_prepended_prompt() # the default one
    logger.debug('Prependend prompt: ' + prepended_prompt)
    prompt_list = [f"PREAMBLE: {prepended_prompt}\nPROMPT: {p}" for p in prompts]
    #prompt_list.append('')

    logger.debug(prompt_list)

    # The target you are pointing to
    objective_target = OpenAIChatTarget(
        endpoint=config_loader._openai_full_endpoint,
        api_key=config_loader._openai_api_key,
        model_name=config_loader._openai_deployment,
        api_version=config_loader._openai_api_version
    )

    #objective_scorer = scorer_factory.get_substring_scorer(substring='feci')
    
    #objective_scorer = scorer_factory.get_float_threshold_azure_content_filter_scorer(
    #   api_key=config_loader._content_filter_key,
    #   full_endpoint=config_loader._content_filter_endpoint,
    #   threshold=0.6)

    objective_scorer=scorer_factory.get_self_ask_refusal_scorer(target_checking_refusal=objective_target)
    # objective_scorer=scorer_factory.get_self_ask_likert_scorer(target=objective_target)

    # Configure the orchestrator you want to use. This is the basis of your attack strategy.
    # This could be a single turn or multi turn.
    # In this case, we're using PromptSendingOrchestrator to simply send the prompts.
    orchestrator = orchestrator_factory.get_prompt_sending_orchestrator(
        target=objective_target,
        scorers=[objective_scorer]
    )

    responses = await orchestrator.send_prompts_async(
        prompt_list=prompt_list,
        memory_labels=memory_labels
    )

    logger.debug('\n\n*** Printing raw responses from target ***\n')
    for resp in responses:
        logger.debug('Direct response:\n' + str(resp.__dict__))

    results = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
    # memory.add_request_pieces_to_memory(request_pieces=results)

    # TODO replace the following with producing a report and saving results to a specified folder
    logger.info('\n\n** Printing flattened response pieces **\n')
    for result in results:
        logger.info('Flattened response piece:\n' + str(result.to_dict()) + '\n********\n')

    score_results = await objective_scorer.score_responses_inferring_tasks_batch_async(request_responses=results)
    memory.add_scores_to_memory(scores=score_results)

    logger.info('\n\n*** Printing score results ***\n')
    for score_res in score_results:
        logger.info('\n**********\n')
        logger.info('Score_result:\n' + str(score_res.to_dict()))

    logger.debug('Orchestrator details: ' + str(orchestrator.__dict__))


if __name__ == "__main__":
    try:
        asyncio.run(run_test_sending_prompts())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
