from pyrit.models import (
    PromptRequestPiece,
    Score
)

from typing import NamedTuple, Optional, Sequence
from collections import namedtuple
from enum import Enum
import json
import uuid

class PromptRequestPieceType(Enum):
    REQUEST = 'request'
    RESPONSE = 'response'
    OTHER = 'other' # this might be system messages (?)

class ReqRespPair(NamedTuple):
    request: PromptRequestPiece
    response: PromptRequestPiece

    def __str__(self):
        # Convert each element via to_dict and pretty-print as JSON
        return json.dumps({'request': self.request.to_dict(), 'response': self.response.to_dict()})

class PromptResult(NamedTuple):
    id: uuid.UUID | str
    original_prompt: str
    converted_prompt: Optional[str]
    # TODO unpack?? or create a dictionary scorer_type: scoring_results?
    scores: Sequence[Score]

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
            scores=req_res_pair.response.scores
        )

    def to_dict(self):
        return json.dumps(
            {
                'id': self.id,
                'original_prompt': self.original_prompt,
                'converted_prompt': self.converted_prompt,
                'score': [s.to_dict() for s in self.scores]
            }
        )

    def __str__(self):
        return self.to_dict()
