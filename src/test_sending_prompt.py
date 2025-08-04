from pyrit.models import PromptRequestResponse
# from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_target import OpenAIChatTarget

import config_loader
import scorer_factory
import orchestrator_factory
from memory_manager import MemoryManager
from dataset_helper import load_dataset

import logging
import asyncio

logger = logging.getLogger(__name__)

config_loader.load_dotenv_with_check()
config_loader.load_openai_configs()

def peek_prompts_and_other_info(seed_prompts_dataset, memory):
    logger.debug(f"Dataset description: {seed_prompts_dataset.description}")
    logger.debug(f"Dataset name: {seed_prompts_dataset.dataset_name}")
    logger.debug(f"Dataset data type: {seed_prompts_dataset.data_type}")
    logger.debug(f"Dataset authors:' + {seed_prompts_dataset.authors}")
    logger.debug(f"Dataset harm categories: {seed_prompts_dataset.harm_categories}")
    logger.debug(f"Seed prompts: {seed_prompts_dataset.get_values()}")

    logger.debug(f"Prompt groups: {seed_prompts_dataset.groups}")
    logger.debug(f"Prompt seed groups loaded in memory: {memory.get_seed_prompt_groups()}")
    for g in memory.get_seed_prompt_groups():
        logger.debug(str(g.prompts))

def get_report(results):
    pass

async def run_test_sending_prompts():
    memory_manager = MemoryManager()
    memory = memory_manager.get_memory()
    # Configure the labels you want to send, these should be unique to this test to make it easier to retrieve
    memory_labels = {"op_name": "new_op", "user_name": "PyRIT_test_framework", "test_name": __file__}

    # Load dataset and prepare prompts
    seed_prompts_dataset = load_dataset(dataset_name='harmbench')
    seed_prompts = seed_prompts_dataset.prompts[:2] # TODO limit the number of prompts for testing!
    await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="pyrit_test_framework")

    peek_prompts_and_other_info(seed_prompts_dataset=seed_prompts_dataset, memory=memory)

    prompts=[p.value for p in seed_prompts]

    system_prompt=config_loader.load_system_prompt() # the default one
    logger.debug(f"System prompt: {system_prompt}")
    prompt_list = [f"SYSTEM PROMPT: {system_prompt}\nMESSAGE PROMPT: {p}" for p in prompts]
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
    logger.debug(f"Orchestrator details: {orchestrator.__dict__}")

    responses = await orchestrator.send_prompts_async(
        prompt_list=prompt_list,
        memory_labels=memory_labels
    )

    orchestrator_req_res_pieces = orchestrator.get_memory()
    # group req/response
    # req/response are matched by conversation_id
    # req have role="user", responses have role="assistant" and prompt_request_response_id filled
    logger.info(f"Orchestrator req/res history:")
    for req_res_piece in orchestrator_req_res_pieces:
        logger.info(f"Req/response piece:\n{req_res_piece.to_dict()}")
    orchestrator_scores = orchestrator.get_score_memory()
    # It's empty, I think for unproper invocation of scorer inside PromptSendingOrchestrator
    logger.debug(f"Orchestrator scores: {orchestrator_scores}")
    orchestrator_id = orchestrator.get_identifier()
    logger.info(f"Orchestrator identifier: {orchestrator_id}")

    logger.info('\n\n*** Printing raw responses from target ***\n')
    for resp in responses:
        logger.info(f"Direct response:\n {resp.__dict__}")
        #memory.add_request_response_to_memory(request=resp)

    results = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
    # NON NEEDED, already added by orchestrator, if trying to do so, there will indeed be primary key violation for duplicate insert (reinserting something already committed to memory)
    #memory.add_request_pieces_to_memory(request_pieces=results)

    # TODO replace the following with producing a report and saving results to a specified folder
    logger.info('\n\n** Printing flattened response pieces **\n')
    for result in results:
        logger.info(f"Flattened response piece:\n {result.to_dict()} \n********\n")

    score_results = await objective_scorer.score_responses_inferring_tasks_batch_async(request_responses=results)
    memory.add_scores_to_memory(scores=score_results)

    logger.info('\n\n*** Printing score results ***\n')
    for score_res in score_results:
        logger.info('\n**********\n')
        logger.info(f"Score_result:\n {score_res.to_dict()}")

    memory_manager.dump_debug_log()
    memory_manager.dump_to_json()

if __name__ == "__main__":
    try:
        asyncio.run(run_test_sending_prompts())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
