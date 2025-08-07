from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter

from typing import Sequence

def get_prompt_sending_orchestrator(
        *,
        target: PromptTarget,
        converters :Sequence[PromptConverter]=None
    ) -> PromptSendingOrchestrator:
    '''
    Enforces the parametrization of the pyrit's PromptSendingOrchestrator.
    We don't want to assign scorers to this orchestrator.
    We manage the scoring separately after all the responses have been received, as some pyrit's
    scorers uses LLM to score the responses; in this way we clearly separate the calls to the LLM to get
    the prompt responses from the calls to the LLM to score these responses.    
    Moreover, we manage here the tuning of the batch size.
    '''
    return PromptSendingOrchestrator(
        objective_target=target,
        prompt_converters=converters,
        batch_size=10 # This is PyRIT default, we can test different values or configure somewhere
    )
