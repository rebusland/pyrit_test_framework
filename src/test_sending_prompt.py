from pyrit.models import PromptRequestResponse, PromptRequestPiece
# from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_target import OpenAIChatTarget

import config_loader
import scorer_factory
import orchestrator_factory
from memory_manager import MemoryManager
from dataset_helper import load_dataset
from logging_handler import logger
import reporting
from data_types import (
    PromptRequestPieceType,
    ReqRespPair,
    PromptResult
)

from collections import defaultdict, namedtuple
import asyncio
from typing import Sequence

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


def get_prompt_request_piece_type(prompt_req_piece: PromptRequestPiece) -> PromptRequestPieceType:
    '''
    Find if a PromptRequestPiece message is a request or a response
    The request pieces are the PromptRequestPiece with "role"="user", "scores" empty, and "sequence"=0
    The response pieces are the PromptRequestPiece with "role"="assistant", "scores" section filled, and "sequence"=1 
    '''
    if prompt_req_piece.role=='user' and prompt_req_piece.sequence==0:
        return PromptRequestPieceType.REQUEST
    elif prompt_req_piece.role=='assistant' and prompt_req_piece.sequence==1:
        return PromptRequestPieceType.RESPONSE
    else:
        return PromptRequestPieceType.OTHER

def group_request_response(req_res_list: Sequence[PromptRequestPiece]) -> Sequence[ReqRespPair]:
    req_res_pair_dict = defaultdict(lambda: {"request": None, "response": None})
    # request and responses pieces are linked through the "conversation_id"
    for req_res_piece in req_res_list:
        prompt_request_type = get_prompt_request_piece_type(req_res_piece)
        if prompt_request_type == PromptRequestPieceType.OTHER:
            logger.warning(f"The following PromptRequestPiece is not a request nor a response:\n{req_res_piece}")
        req_res_pair_dict[req_res_piece.conversation_id][prompt_request_type.value] = req_res_piece

    # ignores others (only request and responses)
    return [
        ReqRespPair(**pair)
        for pair in req_res_pair_dict.values()
        if pair[PromptRequestPieceType.REQUEST.value] and pair[PromptRequestPieceType.RESPONSE.value]
    ]

async def run_test_sending_prompts():
    memory_manager = MemoryManager()
    memory = memory_manager.get_memory()
    # Configure the labels you want to send, these should be unique to this test to make it easier to retrieve
    memory_labels = {"op_name": "new_op", "user_name": "PyRIT_test_framework", "test_name": __file__}

    # Load dataset and prepare prompts
    seed_prompts_dataset = load_dataset(dataset_name='harmbench')
    seed_prompts = seed_prompts_dataset.prompts[:2] # TODO limit the number of prompts for testing!
    for sp in seed_prompts:
        logger.debug(f"Seed prompt: {sp.__dict__}")

    await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="pyrit_test_framework")

    # TODO
    # for some reason the global dataset_name might be empty but set for each prompt
    # write something to infer dataset info from prompts
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

    prompts_metadata_for_orchestrator = {
        'dataset_name': seed_prompts_dataset.name,
        'dataset_harm_categories': seed_prompts_dataset.harm_categories
    }
    responses = await orchestrator.send_prompts_async(
        prompt_list=prompt_list,
        memory_labels=memory_labels,
        metadata=prompts_metadata_for_orchestrator
    )

    orchestrator_id = orchestrator.get_identifier()
    logger.debug(f"Orchestrator identifier: {orchestrator_id}")

    logger.debug('\n\n*** Printing raw responses from target ***\n')
    for resp in responses:
        logger.debug(f"Direct response:\n {resp.__dict__}")

    results = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
    # NON NEEDED, already added by orchestrator, if trying to do so, there will indeed be primary key violation for duplicate insert (reinserting something already committed to memory)
    #memory.add_request_pieces_to_memory(request_pieces=results)

    # TODO replace the following with producing a report and saving results to a specified folder
    logger.debug('\n\n** Printing flattened response pieces **\n')
    for result in results:
        logger.debug(f"Flattened response piece:\n {result.to_dict()} \n********\n")

    score_results = await objective_scorer.score_responses_inferring_tasks_batch_async(request_responses=results)
    memory.add_scores_to_memory(scores=score_results)

    logger.debug('\n\n*** Printing score results ***\n')
    for score_res in score_results:
        logger.debug('\n**********\n')
        logger.debug(f"Score_result:\n {score_res.to_dict()}")

    orchestrator_req_res_pieces = orchestrator.get_memory()
    # group req/response
    # req/response are matched by conversation_id
    # req have role="user", responses have role="assistant" and prompt_request_response_id filled
    logger.debug(f"Orchestrator req/res history after scoring:")
    for req_res_piece in orchestrator_req_res_pieces:
        logger.debug(f"Req/response piece:\n{req_res_piece.to_dict()}")
    orchestrator_scores = [s.to_dict() for s in orchestrator.get_score_memory()]
    # It's empty, I think for unproper invocation of scorer inside PromptSendingOrchestrator
    logger.debug(f"Orchestrator scores after scoring: {orchestrator_scores}")

    # pair each response with its request, using "conversation_id" as linking key
    req_res_pairs = group_request_response(orchestrator_req_res_pieces)
    for req_res_pair in req_res_pairs:
        logger.debug(f"Request/Response pair:\n{req_res_pair}")
    
    prompt_results = [PromptResult.from_req_resp_pair(rr_pair) for rr_pair in req_res_pairs]
    logger.info('***** Printing prompt results ******')
    for p_res in prompt_results:
        logger.info(p_res)

    #reporting.dump_debug_log(memory=memory)
    #reporting.dump_to_json(memory=memory)


if __name__ == "__main__":
    try:
        asyncio.run(run_test_sending_prompts())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
