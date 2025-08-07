import json
from pyrit.memory import MemoryInterface
from pyrit.models import (
    PromptRequestResponse,
    PromptRequestPiece,
    SeedPrompt
)
# from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator
from pyrit.prompt_target import OpenAIChatTarget

import config_loader
import scorer_factory
import scoring_manager
import orchestrator_factory
from memory_manager import MemoryManager
from dataset_helper import load_dataset, peek_dataset_info
from logging_handler import logger
import reporting
from data_types import (
    FattenedScoreResult,
    PromptRequestPieceType,
    ReqRespPair,
    PromptResult,
    ScoresAndResponse
)

from collections import defaultdict, namedtuple
import asyncio
from typing import Sequence
from datetime import datetime

config_loader.load_dotenv_with_check()
config_loader.load_openai_configs()

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

def find_request_and_enrich_score_result(*,
    req_resp_pieces: Sequence[PromptRequestPiece],
    scores_and_response: ScoresAndResponse) -> FattenedScoreResult:
    '''
    Find the request associated to the given response among the list of all the PromptRequestPiece(s) provided
    (some are requests, some are responses, so we have to identify which are requests first).
    request and response pieces are linked through the conversation_id
    '''
    # TODO check if it's present
    conv_id = scores_and_response.prompt_response.conversation_id
    matching_requests = [r for r in req_resp_pieces if get_prompt_request_piece_type(r) == PromptRequestPieceType.REQUEST and r.conversation_id == conv_id]
    # check only one element
    if len(matching_requests) > 1:
        raise ValueError(f"More than one request ({len(matching_requests)}) found for conversation_id {conv_id}")
    elif len(matching_requests) == 0:
        raise ValueError(f"No matching request found for conversation_id {conv_id}")
    
    return FattenedScoreResult(
        prompt_request=matching_requests[0],
        prompt_response=scores_and_response.prompt_response,
        score_or_error=scores_and_response.score_or_error
    )

def group_request_response(req_res_list: Sequence[PromptRequestPiece]) -> Sequence[ReqRespPair]:
    '''
    req_res_pair_dict = defaultdict(lambda: {"request": None, "response": None})
    # request and response pieces are linked through the conversation_id
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
    '''
    pass

def get_prompt_list(seed_prompts: Sequence[SeedPrompt], system_prompt: str=config_loader.EXAMPLE_SAFE_PROMPT) -> Sequence[str]:
    # we remove any newline/cr character
    prompts=[p.value.replace('\n', '').replace('\r', '') for p in seed_prompts]
    system_prompt=config_loader.load_system_prompt(system_prompt) # the default one
    logger.debug(f"System prompt: {system_prompt}")
    # TODO remove SYSTEM PROMPT/MESSAGE PROMPT separation
    prompt_list = [f"SYSTEM PROMPT: {system_prompt} MESSAGE PROMPT: {p}" for p in prompts]
    logger.debug(prompt_list[:3])
    return prompt_list

def peek_prompt_responses(*, direct_responses: Sequence[PromptRequestResponse], flat_responses: Sequence[PromptRequestPiece]) -> None:
    logger.debug('\n\n*** Printing raw responses from target ***\n')
    for resp in direct_responses:
        logger.debug(f"Direct response:\n {resp.__dict__}")
    # not needed, already added by orchestrator, if trying to do so, there will indeed be primary key violation for duplicate insert (reinserting something already committed to memory)
    # memory.add_request_pieces_to_memory(request_pieces=results)
    logger.debug('\n\n** Printing flattened response pieces **\n')
    for flat_res in flat_responses:
        logger.debug(f"Flattened response piece:\n {json.dumps(flat_res.to_dict())} \n********\n")


async def run_test_sending_prompts(dataset_name: str='harmbench'):
    test_name = f"{dataset_name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    logger.info(f"\n\n**** Running test {test_name} ****")
    memory_manager = MemoryManager()
    memory = memory_manager.get_memory()
    # Configure the labels you want to send, these should be unique to this test to make it easier to retrieve
    memory_labels = {
        "op_name": "new_op",
        "user_name": "pyrit_test_framework",
        "dataset_name": dataset_name,
        "test_name": test_name
    }

    ##### LOADING DATASET AND PREPARE PROMPTS #####
    dataset = load_dataset(dataset_name=dataset_name)
    peek_dataset_info(dataset=dataset)

    seed_prompts = dataset.prompts[:5]
    logger.info(f"Prompt dataset {dataset_name} loaded, preparing {len(seed_prompts)} prompts to send")
    await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="pyrit_test_framework")
    prompt_list = get_prompt_list(seed_prompts=seed_prompts)

    ##### SETUP ORCHESTRATOR AND SEND PROMPTS TO TARGET LLM #####
    objective_target = OpenAIChatTarget(
        endpoint=config_loader._openai_full_endpoint,
        api_key=config_loader._openai_api_key,
        model_name=config_loader._openai_deployment,
        api_version=config_loader._openai_api_version
    )

    # We support only single-turn attacks with prompts firing
    orchestrator = orchestrator_factory.get_prompt_sending_orchestrator(target=objective_target)
    logger.debug(f"Orchestrator details: {orchestrator.__dict__}")

    logger.info('Sending prompts to the target')
    # TODO produce a SHA of the prompt to identify it later
    # TODO among metadata trace a correspondance between seed_id (should be a static id for the prompt in the dataset)
    # and the sha of the prompt, so that we can then match the response and the prompt to the original seed_id
    prompts_metadata_for_orchestrator = {
        'dataset_name': dataset.name,
        'dataset_harm_categories': dataset.harm_categories
    }
    responses = await orchestrator.send_prompts_async(
        prompt_list=prompt_list,
        memory_labels=memory_labels,
        metadata=prompts_metadata_for_orchestrator
    )

    flattened_responses = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
    peek_prompt_responses(direct_responses=responses, flat_responses=flattened_responses)

    ######## SCORING ########
    logger.info(f"Scoring the target's responses")
    #objective_scorer = scorer_factory.get_substring_scorer(substring='feci')
    
    #objective_scorer = scorer_factory.get_float_threshold_azure_content_filter_scorer(
    #   api_key=config_loader._content_filter_key,
    #   full_endpoint=config_loader._content_filter_endpoint,
    #   threshold=0.6)

    # TODO to be inverted with TrueFalseInverterScorer
    objective_scorer=scorer_factory.get_self_ask_refusal_scorer(target_checking_refusal=objective_target)
    # objective_scorer=scorer_factory.get_self_ask_likert_scorer(target=objective_target)

    scores_and_responses = await scoring_manager.score_results(scorer=objective_scorer, responses=flattened_responses)    
    # add just valid scores to pyrit memory, valid_scores is a list[list[Score]] and needs to be flattened
    valid_scores = [fsr.score_or_error.unwrap() for fsr in scores_and_responses if fsr.score_or_error.is_success()]
    memory.add_scores_to_memory(scores=[score for sublist in valid_scores for score in sublist])

    logger.debug('**** Score results enriched with the scored prompt response ****')
    for fs in scores_and_responses:
        logger.debug(f"Score + response: {fs}")

    ##### COLLECTING AND MODELLING THE RESULTS #####
    orchestrator_req_res_pieces = orchestrator.get_memory()
    logger.debug(f"Orchestrator req/res history after scoring:")
    for req_res_piece in orchestrator_req_res_pieces:
        logger.debug(f"Req/response piece: {json.dumps(req_res_piece.to_dict())}")

    logger.info('Prompt responses received and scored, gruping requests, responses and scores')
    # pair each response with its request, using "conversation_id" as linking key
    fat_score_results = [find_request_and_enrich_score_result(req_resp_pieces=orchestrator_req_res_pieces, scores_and_response=sr) for sr in scores_and_responses]
    logger.debug(f"Fat req/resp/score results:\n")
    for fsr in fat_score_results:
        logger.debug(f"Fat req/resp/score: {fsr}")

    '''
    logger.info('Prompt responses received and scored, gruping requests and responses')
    # pair each response with its request, using "conversation_id" as linking key
    req_res_pairs = group_request_response(orchestrator_req_res_pieces)
    for req_res_pair in req_res_pairs:
        logger.debug(f"Request/Response pair:\n{req_res_pair}")
    
    prompt_results = [PromptResult.from_req_resp_pair(rr_pair) for rr_pair in req_res_pairs]
    logger.debug('***** Printing prompt results ******')
    for p_res in prompt_results:
        logger.debug(p_res)

    ##### SAVE AND REPORT TEST RESULTS #####
    logger.info('Saving prompt results and producing test report')
    reporting.save_prompt_results_to_csv(results=prompt_results, compact=True, test_name=test_name)
    #reporting.dump_debug_log(memory=memory)
    #reporting.dump_to_json(memory=memory)

    logger.info(f"**** Finished test {test_name} ****")
    '''

if __name__ == "__main__":
    try:
        asyncio.run(run_test_sending_prompts())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
