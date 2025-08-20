from typing import Sequence
from pyrit.memory import MemoryInterface

from data_types import (
    PromptResult
)
from logging_handler import logging
from config_handler import load_test_config

from datetime import datetime
import os
from pathlib import Path
import csv
import json

# Use pathlib.Path to handle paths in an OS-independent way
_OUTPUTS_DIR = Path(load_test_config()['output']['dir'])
# Create output directory if it doesn't exist
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_FILE_PROMPT_RESULT_PREFIX='prompt_results'

def save_prompt_results_to_csv(
        *,
        results: Sequence[PromptResult],
        compact: bool=False,
        output_folder_path: Path=_OUTPUTS_DIR,
        test_name: str=f"test_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    ) -> None:
    if not results:
        print("No data to write.")
        return

    result_mapper = PromptResult.to_dict_reduced_and_try_scores_flattening if compact else PromptResult.to_dict_extended
    dict_rows = [result_mapper(result) for result in results]
    file_path=output_folder_path / f"{_FILE_PROMPT_RESULT_PREFIX}_{test_name}.csv"

    # Write to CSV
    with file_path.open(mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_rows[0].keys())
        writer.writeheader()
        writer.writerows(dict_rows)

def read_results_from_csv(*, file_name: str, compact: bool=False):
    pass

def get_test_summary_report(*, results: Sequence[PromptResult]) -> dict:
    '''
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
    pass

def dump_to_json(*, memory: MemoryInterface) -> None:
    memory.export_conversations(file_path=Path(_OUTPUTS_DIR)/"conversation.json")

def dump_debug_log(*, memory: MemoryInterface, log_name: str="debug.log", orchestrator_id=None) -> None:
    '''
    log_file = os.path.join(OUTPUTS_DIR, log_name)
    with open(log_file, "w", encoding="utf-8") as f:
        element = memory.get_prompt_request_pieces(orchestrator_id=orchestrator_id)
        for r in element:
            f.write("\n" + "*"*100 + "\n")
            f.write(str(r.to_dict()))

    if not DEBUG_LOGGING:
        return

    log_file = os.path.join(MEMORY_RESULTS_DIR, log_name)
    prompt_request_pieces = self._memory.get_prompt_request_pieces()
    with open(log_file, "w", encoding="utf-8") as f:
        for rr in prompt_request_pieces:
            f.write(f"--- REQUEST/RESPONSE ID: {rr.id} ---\n")
            f.write(f"Original prompt id: {rr.original_prompt_id} ---\n")
            f.write(f"Original value data type: {rr.original_value_data_type} ---\n")
            f.write(f"Prompt request response value:\n{rr.original_value}\n")
            for s in self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[rr.original_prompt_id]):
                if s.id == rr.original_prompt_id:
                    f.write(f"""Score Rationale: {s.score_rationale}\n 
                        Score Value: {s.score_value}\n
                        Score category: {s.score_category}""") #etc. etc.
            f.write("\n")
    '''
    pass
