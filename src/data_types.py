from pyrit.models import (
    PromptRequestPiece,
    Score,
    ScoreType
)

from typing import Any, Callable, NamedTuple, Optional, Sequence
from collections import namedtuple
from enum import Enum
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class PromptRequestPieceType(Enum):
    REQUEST = 'request'
    RESPONSE = 'response'
    OTHER = 'other' # this might be system messages (?)

class Scorer(Enum):
    """
    The scorers we support from PyRIT: 
    """
    SelfAskRefusalScorer='SelfAskRefusalScorer'

@dataclass
class ScoresOrError:
    '''
    Pyrit scorers return a list of scores for a single evaluation. In most cases, it's a single score,
    but in some cases different scores are produced for a single prompt response.
    This might store also an error message if scoring failed.
    '''
    scores: Optional[Sequence[Score]] = None
    error: Optional[Exception] = None

    @staticmethod
    def from_scores(scores: Sequence[Score]) -> 'ScoresOrError':
        return ScoresOrError(scores=scores)

    @staticmethod
    def from_error(error: Exception) -> 'ScoresOrError':
        return ScoresOrError(error=error)

    def is_success(self) -> bool:
        return self.error is None

    def is_error(self) -> bool:
        return self.error is not None

    def unwrap(self) -> Sequence[Score]:
        if self.error:
            raise self.error
        if self.scores is None:
            raise ValueError("No scores present despite success status")
        return self.scores

    def to_dict(self):
        if self.error:
            return {
                "error": self.error.__dict__
            }

        if self.scores is None:
            raise ValueError("No scores present despite missing errors")

        return {
            "scores": [s.to_dict() for s in self.scores]
        }

    def __str__(self):
        return json.dumps(self.to_dict())

@dataclass
class ScoresAndResponse:
    '''
    It stores scores result and the response that was scored.
    '''
    prompt_response: PromptRequestPiece
    score_or_error: ScoresOrError

    def to_dict(self):
        return {
            "prompt_response": self.prompt_response.to_dict(),
            "score_or_error": self.score_or_error.to_dict()
        }

    def __str__(self):
        return json.dumps(self.to_dict())

@dataclass
class FattenedScoringResult:
    '''
    It stores also info about the prompt request sent and the response that was scored.
    '''
    prompt_request: PromptRequestPiece
    prompt_response: PromptRequestPiece
    score_or_error: ScoresOrError

    def to_dict(self):
        return {
            "prompt_request": self.prompt_request.to_dict(),
            "prompt_response": self.prompt_response.to_dict(),
            "score_or_error": self.score_or_error.to_dict()
        }

    def __str__(self):
        return json.dumps(self.to_dict())

@dataclass
class CompactScoreResult:
    '''
    A more "flattened" and simplified version of the pyrit Score class.add.
    '''
    score_value: bool
    score_value_description: str
    score_type: ScoreType
    score_category: str
    score_rationale: str
    score_timestamp: datetime
    scorer: Scorer

    @staticmethod
    def from_valid_score(score: Score) -> 'CompactScoreResult':
        return CompactScoreResult(
            score_value=score.score_value,
            score_value_description=score.score_value_description,
            score_type=score.score_type,
            score_category=score.score_category,
            score_rationale=score.score_rationale,
            score_timestamp=score.timestamp,
            scorer=Scorer(score.scorer_class_identifier['__type__'])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_value": self.score_value,
            "score_value_description": self.score_value_description,
            "score_type": self.score_type,
            "score_category": self.score_category,
            "score_rationale": self.score_rationale,
            "score_timestamp": self.score_timestamp.isoformat(),
            "scorer": self.scorer.value
        }

    def __str__(self):
        return json.dumps(self.to_dict())

class PromptResult(NamedTuple):
    id: uuid.UUID | str
    original_prompt: str
    converted_prompt: Optional[str]
    target_response: Optional[str]
    # TODO unpack?? or create a dictionary scorer_type: scoring_results?
    scores_or_error: ScoresOrError

    @staticmethod
    def from_fat_score_result(fat_scoring_res: FattenedScoringResult) -> 'PromptResult':
        '''
        Extract relevant infos from the req/resp/scores (original prompt, target response, scoring metrics etc.)
        From the request piece we have these relevant info:
        - under "original_value" the original prompt
        - under "converted_value" the prompt modified by the converter(s)
        - "response_error" might be useful

        From the response pieces we can get these relevant info:
        - under "scores"[index]."task" I have the original prompt sent (NOT CONVERTED)
        - under "original_value" the response from the target
        - under "scores" all the relevant scoring infos (score_rationale, score_value etc.) for each scorer applied
        - We then have other useful info like "converted_value"
        '''
        return PromptResult(
            id=fat_scoring_res.prompt_request.conversation_id,
            original_prompt=fat_scoring_res.prompt_request.original_value,
            converted_prompt=fat_scoring_res.prompt_request.converted_value,
            target_response=fat_scoring_res.prompt_response.original_value,
            scores_or_error=fat_scoring_res.score_or_error
        )

    def to_dict(self, *,
            extended: bool=True,
            scores_error_mapper: Callable[[ScoresOrError], Sequence[Any] | dict] = lambda s: s.to_dict()
        ) -> dict[str, Any]:

        out = {
            "id": self.id,
            "original_prompt": self.original_prompt,
            "scores": scores_error_mapper(self.scores_or_error)
        }
        if extended:
            out["converted_prompt"] = self.converted_prompt
            out["target_response"] = self.target_response

        return out

    def to_dict_extended(self) -> dict[str, Any]:
        return self.to_dict(extended=True)

    def to_dict_reduced(self) -> dict[str, Any]:
        return self.to_dict(
            extended=False,
            scores_error_mapper=lambda scores_or_error: 
            [CompactScoreResult.from_valid_score(s).to_dict() for s in scores_or_error.unwrap()] if scores_or_error.is_success() else scores_or_error.to_dict()                
        )

    def to_dict_reduced_and_try_scores_flattening(self) -> dict[str, Any]:
        '''
        Check if scores is actually one element (typical): if so the dictionary representation of scores is replaced with a plain dict instead of a list
        '''
        if self.scores_or_error.is_success() and len(self.scores_or_error.unwrap()) == 1:
            return self.to_dict(extended=False,
                scores_error_mapper= lambda sc_err : CompactScoreResult.from_valid_score(sc_err.unwrap()[0]).to_dict())
        
        return self.to_dict_reduced()

    def __str__(self):
        return json.dumps(self.to_dict())
