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
    FattenedScoringResult,
    PromptRequestPieceType,
    PromptResult,
    ScoresAndResponse
)
from utils import run_only_if_log_level_debug

import asyncio
from typing import Any, Sequence
from datetime import datetime
import json
import itertools

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
    scores_and_response: ScoresAndResponse) -> FattenedScoringResult:
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
    
    return FattenedScoringResult(
        prompt_request=matching_requests[0],
        prompt_response=scores_and_response.prompt_response,
        score_or_error=scores_and_response.score_or_error
    )

def get_prompt_list(seed_prompts: Sequence[SeedPrompt], system_prompt: str=config_loader.EXAMPLE_SAFE_PROMPT) -> Sequence[str]:
    # we remove any newline/cr character
    prompts=[p.value.replace('\n', '').replace('\r', '') for p in seed_prompts]
    system_prompt=config_loader.load_system_prompt(system_prompt) # the default one
    logger.debug(f"System prompt: {system_prompt}")
    # TODO remove SYSTEM PROMPT/MESSAGE PROMPT separation
    prompt_list = [f"SYSTEM PROMPT: {system_prompt} MESSAGE PROMPT: {p}" for p in prompts]
    logger.debug(f"Peek the first three prompts to send:\n{prompt_list[:3]}")
    return prompt_list

@run_only_if_log_level_debug()
def peek_scores_in_memory(*, memory: MemoryInterface, scores_and_responses: Sequence[ScoresAndResponse]):
    valid_scores = list(itertools.chain.from_iterable([fsr.score_or_error.unwrap() for fsr in scores_and_responses if fsr.score_or_error.is_success()]))
    scores_in_memory = memory.get_scores_by_prompt_ids(prompt_request_response_ids=[vs.prompt_request_response_id for vs in valid_scores])  # this is not working: get_scores_by_orchestrator_id(orchestrator_id=orchestrator.get_identifier())
    peek_iterable(iterable=scores_in_memory, header=f"There are {len(scores_in_memory)} Score objects committed to pyrit memory", element_description="Score in pyrit memory", stringifyier=lambda score_in_memory : json.dumps(score_in_memory.to_dict()))

@run_only_if_log_level_debug()
def peek_iterable(
    *,
    iterable: Sequence[Any],
    header: str,
    element_description: str,
    stringifyier = lambda el : str(el)
) -> None:
    logger.debug(f"\n\n\n ****** {header} ******\n\n")
    for el in iterable:
        logger.debug(f"{element_description}: {stringifyier(el)}")

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

    seed_prompts = dataset.prompts[:2]
    logger.info(f"\n\n\nPrompt dataset {dataset_name} loaded, preparing {len(seed_prompts)} prompts to send\n")
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

    peek_iterable(iterable=responses, header=f"Printing {len(responses)} raw responses from target", element_description='Direct response', stringifyier=lambda el : str(el.__dict__))
    peek_iterable(iterable=flattened_responses, header=f"Printing {len(flattened_responses)} flattened response pieces", element_description='Flattened response piece', stringifyier=lambda el : json.dumps(el.to_dict()))
    # not needed, already added by orchestrator, if trying to do so, there will indeed be primary key violation for duplicate insert (reinserting something already committed to memory)
    # memory.add_request_pieces_to_memory(request_pieces=results)

    ######## SCORING ########
    logger.info(f"Scoring the target's responses")    
    #objective_scorer = scorer_factory.get_float_threshold_azure_content_filter_scorer(
    #   api_key=config_loader._content_filter_key,
    #   full_endpoint=config_loader._content_filter_endpoint,
    #   threshold=0.6)

    # TODO to be inverted with TrueFalseInverterScorer
    objective_scorer=scorer_factory.get_self_ask_refusal_scorer(target_checking_refusal=objective_target)
    # objective_scorer=scorer_factory.get_self_ask_likert_scorer(target=objective_target)

    scores_and_responses = await scoring_manager.score_results(
        scorer=objective_scorer,
        responses=flattened_responses,
        memory=memory
    )
    peek_iterable(iterable=scores_and_responses, header="Score results enriched with the scored prompt response", element_description="Score + Response")
    peek_scores_in_memory(memory=memory, scores_and_responses=scores_and_responses)

    ##### COLLECTING AND MODELLING THE RESULTS #####
    orchestrator_req_res_pieces = orchestrator.get_memory()
    peek_iterable(iterable=orchestrator_req_res_pieces, header=f"Orchestrator req/res history after scoring", element_description="Req/response piece", stringifyier=lambda req_res_piece : json.dumps(req_res_piece.to_dict()))

    logger.info('Prompt responses received and scored, gruping requests, responses and scores')
    # pair each response with its request, using "conversation_id" as linking key
    fat_score_results = [find_request_and_enrich_score_result(req_resp_pieces=orchestrator_req_res_pieces, scores_and_response=sr) for sr in scores_and_responses]
    peek_iterable(iterable=fat_score_results, header="Fat req/resp/score results", element_description="Fat req/resp/score")

    prompt_results = [PromptResult.from_fat_score_result(fsr) for fsr in fat_score_results]
    peek_iterable(iterable=prompt_results, header="Prompt results", element_description="Prompt result")

    ##### SAVE AND REPORT TEST RESULTS #####
    logger.info('Saving prompt results and producing test report')
    reporting.save_prompt_results_to_csv(results=prompt_results, compact=True, test_name=test_name)
    #reporting.dump_debug_log(memory=memory)
    #reporting.dump_to_json(memory=memory)

    logger.info(f"**** Finished test {test_name} ****")


if __name__ == "__main__":
    try:
        asyncio.run(run_test_sending_prompts())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
