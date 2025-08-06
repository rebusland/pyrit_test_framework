from pyrit.orchestrator import PromptSendingOrchestrator

def get_prompt_sending_orchestrator(*, target, converters=None, scorers=[]):
    return PromptSendingOrchestrator(
        objective_target=target,
        prompt_converters=converters,
        scorers=scorers,
        batch_size=10 # This is PyRIT default, we can test different values or configure somewhere
    )
