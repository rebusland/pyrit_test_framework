from typing import Dict, Sequence
from pyrit.memory import MemoryInterface

from data_types import (
    ReqRespPair,
    PromptResult
)
from logging_handler import logging

import os
import pathlib
import csv
import json

# TODO move and retrive from a config file!!
# Where to save reports, conversations and so on
OUTPUTS_DIR = "results/"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def save_prompt_results_to_csv(*, results: Sequence[PromptResult], output_folder_path: str=OUTPUTS_DIR, file_name: str='test_results.csv') -> None:
    '''
    with open(output_folder_path + '/' + file_name, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=filtered_data[0].keys())
        writer.writeheader()
        writer.writerows(filtered_data)
    '''
    pass

def get_test_summary_report(*, results: Sequence[ReqRespPair]) -> dict:
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
    memory.export_conversations(file_path=pathlib.Path(OUTPUTS_DIR)/"conversation.json")

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
