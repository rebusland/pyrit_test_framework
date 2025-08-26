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
        flat: bool=True, # slim business representation
        compact: bool=False, # it has effect only if flat=False
        results_subfolder: str='',
        test_name: str=f"test_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    ) -> None:
    '''
    flat: bool -> produces a slim report with just what's needed for business.
    compact: bool -> whether results should be complete (extended) or reduced. It has effect only if flat=False. To be use for getting "under the hood" info of each prompt evaluation.
    '''
    if not results:
        print("No data to write.")
        return

    dict_rows = {}
    if flat:
        dict_rows = [result.to_dict_flat() for result in results]
    else:
        result_mapper = PromptResult.to_dict_reduced if compact else PromptResult.to_dict_extended
        dict_rows = [result_mapper(result) for result in results]

        # Convert JSON-like fields into proper JSON strings
        for row in dict_rows:
            if isinstance(row.get("scores"), (dict, list)):
                row["scores"] = json.dumps(row["scores"], ensure_ascii=False)

    base_dir = _OUTPUTS_DIR / results_subfolder if results_subfolder else _OUTPUTS_DIR
    # Create output directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path= base_dir / f"{_FILE_PROMPT_RESULT_PREFIX}_{test_name}.csv"

    # Write to CSV
    with file_path.open(mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_rows[0].keys())
        writer.writeheader()
        writer.writerows(dict_rows)

def load_results_from_csv(*,
    file_name: str,
    compact: bool=False,
    output_subfolder: str,
    ) -> Sequence[PromptResult]:
    f_suffix = Path(file_name).suffix
    if f_suffix and f_suffix != '.csv':
        raise ValueError(f"{f_suffix} extension is not valid. Only results from csv files can be loaded")
    fname = file_name if f_suffix else f"{file_name}.csv"
    out_folder_path = _OUTPUTS_DIR if not output_subfolder else _OUTPUTS_DIR / output_subfolder 
    file_path= out_folder_path / fname
    
    results: Sequence[PromptResult] = []
    with file_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(PromptResult.from_dict(row, compact=compact))
    return results

def get_test_summary_report(*, results: Sequence[PromptResult]) -> dict:
    '''
    from the request piece I have these relevant info:
     - under "original_value" the original prompt
     - "response_error" might be useful

    from the response pieces I can get these relevant info:
     - under "scores"[index]."task" I have the original prompt sent (NOT CONVERTED)
     - under "original_value" the response from the target
     - under "scores" all the relevant scoring infos (score_rationale, score_value etc.) for each scorer applied
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
