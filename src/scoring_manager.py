from pyrit.exceptions import InvalidJsonException
from pyrit.models import (
    PromptRequestPiece,
    Score
)
from pyrit.score import Scorer

from logging_handler import logger
from data_types import ScoresAndResponse, ScoresOrError
from typing import Sequence
import json

async def score_results(scorer: Scorer, responses: Sequence[PromptRequestPiece]) -> Sequence[ScoresAndResponse]:
    '''
    Score each response at a time:
    e.g. when scoring using an LLM we can tune the delay between scoring evaluation
    and/or handle score exceptions one at a time.
    The "self ask" scorers might ask a LLM to score a response, but this scoring request itself could be interpreted as a jailbreak attack,
    so a content filter error is returned and so the json of the scoring response is not valid for the scorer to be used, hence InvalidJsonException is fired.
    '''
    fat_score_results = []
    for resp in responses:
        try:
            score_r = await scorer.score_async(resp)
            fat_score_results.append(
                ScoresAndResponse(
                    score_or_error=ScoresOrError.from_scores(score_r),
                    prompt_response=resp
                )
            )

        except InvalidJsonException as e:
            fat_score_results.append(
                ScoresAndResponse(
                    score_or_error=ScoresOrError.from_error(e),
                    prompt_response=resp
                )
            )
            logger.warning(f"Unable to score the following prompt response {json.dumps(resp.to_dict())}", stack_info=True, exc_info=True)

        except Exception as e:
            logger.error(f"Unhandled exception from scoring process: {e}", exc_info=True)
            raise

    return fat_score_results

async def score_results_chunk(scorer: Scorer, responses: Sequence[PromptRequestPiece]) -> Sequence[Score]:
    score_results = await scorer.score_responses_inferring_tasks_batch_async(
        request_responses=responses,
        # pyrit's default is 10
        # we use batch_size of 1 to avoid reaching the endpoint throughput limit.
        # we may try different configs (and load this from a conf file)
        batch_size=1
    )

    logger.debug('\n\n*** Printing score results ***\n')
    for score_res in score_results:
        logger.debug('\n**********\n')
        logger.debug(f"Score_result:\n {score_res.to_dict()}")

    return score_results
