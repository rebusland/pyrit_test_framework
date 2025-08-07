from pyrit.models import (
    PromptRequestPiece,
    Score,
    ScoreType
)

from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence
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

class ReqRespPair(NamedTuple):
    request: PromptRequestPiece
    response: PromptRequestPiece

    def __str__(self):
        # Convert each element via to_dict and pretty-print as JSON
        return json.dumps({
            'request': self.request.to_dict(),
            'response': self.response.to_dict()
        })

class Scorer(Enum):
    """
    The scorers we support from PyRIT: 
    """
    SelfAskRefusalScorer='SelfAskRefusalScorer'

@dataclass
class ScoresOrError:
    '''
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
class FattenedScoreResult:
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
class FlatScoreResult:
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
    def from_valid_score(score: Score) -> 'FlatScoreResult':
        return FlatScoreResult(
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
            'score_value': self.score_value,
            'score_value_description': self.score_value_description,
            'score_type': self.score_type,
            'score_category': self.score_category,
            'score_rationale': self.score_rationale,
            'score_timestamp': self.score_timestamp.isoformat(),
            'scorer': self.scorer.value
        }

class PromptResult(NamedTuple):
    id: uuid.UUID | str
    original_prompt: str
    converted_prompt: Optional[str]
    target_response: Optional[str]
    # TODO unpack?? or create a dictionary scorer_type: scoring_results?
    scores: Sequence[Score]
    #scores: Sequence[ScoreResult]

    @staticmethod
    def from_req_resp_pair(req_res_pair: ReqRespPair) -> 'PromptResult':
        '''
        Extract relevant infos from the Request/Response pair (original prompt, target response, scoring metrics etc.)
        from the request piece I have these relevant info:
        - under "original_value" the original prompt
        - under "converted_value" the prompt modified by the converter(s)
        - "response_error" might be useful

        from the response pieces I can get these relevant info:
        - under "scores"[index]."task" I have the original prompt sent (NOT CONVERTED)
        - under "original_value" the response from the target
        - under "scores" all the relevant scoring infos (score_rationale, score_value etc.) for each scorer applied
        - I have then other useful info like "converted_value"
        '''
        return PromptResult(
            id=req_res_pair.request.conversation_id,
            original_prompt=req_res_pair.request.original_value,
            converted_prompt=req_res_pair.request.converted_value,
            target_response=req_res_pair.response.original_value,
            scores=req_res_pair.response.scores
        )

    def to_dict(self, *,
            extended: bool=True,
            score_mapper: Callable[[Score], Any] = lambda s: json.dumps(s.to_dict())
        ) -> dict[str, Any]:

        out = {
            'id': self.id,
            'original_prompt': self.original_prompt,
            'scores': [score_mapper(s) for s in self.scores]
        }
        if extended:
            out['converted_prompt'] = self.converted_prompt
            out['target_response'] = self.target_response

        return out

    def to_dict_extended(self) -> dict[str, Any]:
        return self.to_dict(extended=True)

    def to_dict_reduced(self) -> dict[str, Any]:
        return self.to_dict(extended=False, score_mapper=lambda s: ScoreResult.from_score(s).to_dict())

    def __str__(self):
        return json.dumps(self.to_dict())
