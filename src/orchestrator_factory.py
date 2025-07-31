from pyrit.orchestrator import PromptSendingOrchestrator

def get_prompt_sending_orchestrator(*, target, converters=None, scorers=[]):
    return PromptSendingOrchestrator(
        objective_target=target,
        prompt_converters=converters,
        scorers=scorers
    )

### TODO how to factory class?? ###