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
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

import config_handler
import utils

class PromptRequestPieceType(Enum):
    REQUEST = 'request'
    RESPONSE = 'response'
    OTHER = 'other' # this might be system messages (?)

def score_from_dict(input_dict: Dict[str, Any]) -> Score:
    """
    Deserialize a dictionary (parsed from JSON) into a Score object.
    """
    # Normalize ID
    score_id = input_dict.get("id")
    if score_id:
        try:
            score_id = uuid.UUID(str(score_id))
        except Exception:
            score_id = str(score_id)

    # Normalize prompt_request_response_id
    prr_id = input_dict.get("prompt_request_response_id")
    if prr_id:
        try:
            prr_id = uuid.UUID(str(prr_id))
        except Exception:
            prr_id = str(prr_id)

    # Normalize timestamp
    ts = input_dict.get("timestamp")
    if ts:
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            ts = None

    # Ensure metadata can be None or dict -> str
    metadata = input_dict.get("score_metadata")
    if metadata is None:
        metadata = ""

    # Ensure scorer_class_identifier is always dict
    sci = input_dict.get("scorer_class_identifier") or {}

    # Normalize score_value: must be string because Score.validate expects "true"/"false" or float
    score_value = str(input_dict.get("score_value"))

    # Build Score
    return Score(
        id=score_id,
        score_value=score_value,
        score_value_description=input_dict.get("score_value_description", ""),
        score_type=input_dict.get("score_type"),
        score_category=input_dict.get("score_category", ""),
        score_rationale=input_dict.get("score_rationale", ""),
        score_metadata=metadata,
        scorer_class_identifier=sci,
        prompt_request_response_id=prr_id,
        timestamp=ts,
        task=input_dict.get("task")
    )

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

    @staticmethod
    def from_dict(d: dict[str, Any], compact: bool=False) -> "ScoresOrError":
        if "error" in d:
            # Try to rebuild the error as a generic Exception with its message
            err_dict = d["error"]
            msg = err_dict.get("args", ["Unknown error"])[0] if isinstance(err_dict, dict) else str(err_dict)
            return ScoresOrError.from_error(Exception(msg))

        if "scores" in d:
            scores=[]
            if compact:
                scores = [CompactScoreResult.from_dict(s).to_score() for s in d["scores"]]
            else:
                scores = [score_from_dict(s) for s in d["scores"]]
            return ScoresOrError.from_scores(scores)

        raise ValueError(f"Invalid dictionary for ScoresOrError: {d}")

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
    A more "flattened" and simplified version of the pyrit Score class.
    '''
    score_value: bool
    score_value_description: str
    score_type: ScoreType
    score_category: str
    score_rationale: str
    score_timestamp: datetime
    scorer: str

    @staticmethod
    def from_valid_score(score: Score) -> 'CompactScoreResult':
        return CompactScoreResult(
            score_value=score.score_value,
            score_value_description=score.score_value_description,
            score_type=score.score_type,
            score_category=score.score_category,
            score_rationale=score.score_rationale,
            score_timestamp=score.timestamp,
            scorer=score.scorer_class_identifier['__type__']
        )

    def to_score(self) -> Score:
        '''
        Uses the partial information in CompactScoreResult to build as best as it can a pyrit Score object.
        '''
        return Score(
            id='', # not available in CompactScoreResult
            score_value=self.score_value,
            score_value_description=self.score_value_description,
            score_type=self.score_type,
            score_category=self.score_category,
            score_rationale=self.score_rationale,
            score_metadata='', # not available in CompactScoreResult
            scorer_class_identifier={"__type__": self.scorer},
            prompt_request_response_id='', # not available in CompactScoreResult
        )

    @staticmethod
    def from_dict(d: dict) -> "CompactScoreResult":
        return CompactScoreResult(
            score_value=bool(d["score_value"]),
            score_value_description=d["score_value_description"],
            score_type=d["score_type"],
            score_category=d["score_category"],
            score_rationale=d["score_rationale"],
            score_timestamp=datetime.fromisoformat(d["score_timestamp"]) if isinstance(d["score_timestamp"], str) else d["score_timestamp"],
            scorer=d["scorer"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_value": self.score_value,
            "score_value_description": self.score_value_description,
            "score_type": self.score_type,
            "score_category": self.score_category,
            "score_rationale": self.score_rationale,
            "score_timestamp": self.score_timestamp.isoformat(),
            "scorer": self.scorer
        }

    def __str__(self):
        return json.dumps(self.to_dict())

class PromptResult(NamedTuple):
    id: uuid.UUID | str
    original_prompt: str
    target_response: Optional[str]
    # TODO unpack?? or create a dictionary scorer_type: scoring_results?
    scores_or_error: ScoresOrError

    @staticmethod
    def from_fat_score_result(fat_scoring_res: FattenedScoringResult) -> 'PromptResult':
        '''
        Extract relevant infos from the req/resp/scores (original prompt, target response, scoring metrics etc.)
        From the request piece we have these relevant info:
        - under "original_value" the original prompt
        - "response_error" might be useful

        From the response pieces we can get these relevant info:
        - under "scores"[index]."task" I have the original prompt sent (NOT CONVERTED)
        - under "original_value" the response from the target
        - under "scores" all the relevant scoring infos (score_rationale, score_value etc.) for each scorer applied
        '''
        return PromptResult(
            id=fat_scoring_res.prompt_request.conversation_id,
            original_prompt=fat_scoring_res.prompt_request.original_value,
            target_response=fat_scoring_res.prompt_response.original_value,
            scores_or_error=fat_scoring_res.score_or_error
        )

    @staticmethod
    def from_dict(row: dict[str, str], compact: bool=False) -> "PromptResult":
        """
        Convert a CSV row (as dict) into a PromptResult.
        """
        return PromptResult(
            id=row["id"],
            original_prompt=row["original_prompt"],
            target_response=row.get("target_response"),
            scores_or_error=ScoresOrError.from_dict(json.loads(row["scores"].strip()), compact=compact)
        )

    def to_dict(self, *,
            extended: bool=True,
            scores_error_mapper: Callable[[ScoresOrError], dict] = lambda s: s.to_dict()
        ) -> dict[str, Any]:

        out = {
            "id": self.id,
            "original_prompt": self.original_prompt,
            "scores": scores_error_mapper(self.scores_or_error)
        }
        if extended:
            out["target_response"] = self.target_response

        return out

    def to_dict_extended(self) -> dict[str, Any]:
        return self.to_dict(extended=True)

    def to_dict_reduced(self) -> dict[str, Any]:
        def scores_error_mapper(s_e: ScoresOrError):
            if s_e.is_error():
                return {"error": s_e.to_dict()}
            return {"scores": [CompactScoreResult.from_valid_score(s).to_dict() for s in s_e.unwrap()]}

        return self.to_dict(
            extended=False,
            scores_error_mapper=scores_error_mapper
        )

    def to_dict_reduced_and_try_scores_flattening(self) -> dict[str, Any]:
        '''
        TODO add complexity. Maybe it's better to deprecate (not uniform serialization/deserialization if single or multiple scores)
        Check if scores is actually one element (typical): if so the dictionary representation of scores is replaced with a plain dict instead of a list
        '''
        if self.scores_or_error.is_success() and len(self.scores_or_error.unwrap()) == 1:
            return self.to_dict(
                extended=False,
                scores_error_mapper= lambda sc_err : {"scores": CompactScoreResult.from_valid_score(sc_err.unwrap()[0]).to_dict()}
            )
        
        return self.to_dict_reduced()

    def to_dict_flat(self) -> Dict[str, Any]:
        '''
        Just what's needed for business, useful for producing neat csv results to filter in excel.
        If the score was an error, None is set to all the fields.
        NB the scores results for each request are assumed to be one (it should always be like so)
        and the scoring results (if valid) are then flattened.
        '''
        valid_score = self.scores_or_error.unwrap()[0] if self.scores_or_error.is_success() else None
        return {
            "id": self.id,
            "original_prompt": self.original_prompt,
            "score_value": valid_score.score_value if valid_score else None,
            "score_rationale": valid_score.score_rationale if valid_score else None,
            "score_timestamp": valid_score.timestamp.isoformat() if valid_score else None
        }

    def __str__(self):
        return json.dumps(self.to_dict())


@dataclass
class SingleTestSummary:
    '''
    Synthetic summary for the evaluation of a single dataset.
    - single dataset test name
    - number of prompts fired
    - total number of tokens fired (total number of words)
    - number of jailbreaks
    - percentage jailbreaks
    - number of response errors (prompt response receival or response scoring failed for technical reasons)
    - time taken to evaluate the dataset
    '''
    test_label: str
    num_prompts: int
    num_tokens: int
    num_jailbreaks: int
    perc_jailbreaks: float
    num_response_error: int
    elapsed: str # use utils.format_duration to convert from seconds to nice str representation

    def __init__(self, *, results: Sequence[PromptResult], label: str = 'single_label', elapsed: float=0.):
        self.test_label = label
        self.num_prompts = len(results)
        self.num_tokens = sum([utils.num_tokens_for_model(text=r.original_prompt, model_name=config_handler.get_model_deployment()) for r in results])
        from scoring_manager import is_jailbreak
        self.num_jailbreaks = sum([1 for r in results if r.scores_or_error.is_success() and is_jailbreak(r.scores_or_error.scores[0])])
        self.perc_jailbreaks = self.num_jailbreaks / sum([1 for r in results if r.scores_or_error.is_success()])
        self.num_response_error = sum([1 for r in results if r.scores_or_error.is_error()])
        self.elapsed = utils.format_duration(elapsed)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))

    def __str__(self):
        return self.to_json()

@dataclass
class CompositeTestSummary:
    '''
    The full summary report for the complete test (multiple datasets)
    '''
    single_summaries: Sequence[SingleTestSummary]
    composite_summary: SingleTestSummary # The "summary of the summaries"

    def __init__(self, single_summaries: Sequence[SingleTestSummary]):
        if not single_summaries:
            raise ValueError('No list of summaries was provided to build a composite summary')
        self.single_summaries = single_summaries

        if len(single_summaries) == 1:
            self.composite_summary = single_summaries[0]
        else:
            # ... TODO
            self.composite_summary = single_summaries[0]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))

    def __str__(self):
        return self.to_json()