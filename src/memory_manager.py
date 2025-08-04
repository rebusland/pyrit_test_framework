# memory_manager.py

from pyrit.common import (
    initialize_pyrit,
    IN_MEMORY,
    DUCK_DB
)
from pyrit.memory import MemoryInterface
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestResponse
from config_loader import OUTPUTS_DIR

import pathlib
import pandas as pd
from datetime import datetime
from typing import Any, Optional
import os

# TODO move and retrive from a config file!!
MEMORY_RESULTS_DIR = OUTPUTS_DIR + "memory"
os.makedirs(MEMORY_RESULTS_DIR, exist_ok=True)

MEMORY_MODE = IN_MEMORY # os.getenv("MEMORY_MODE", "DUCK_DB").upper()
DEBUG_LOGGING = True # os.getenv("DEBUG_LOG", "false").lower() == "true"
DUCK_DB_PATH = os.path.join(MEMORY_RESULTS_DIR, "pyrit_results.db")

class MemoryManager:
    _instance = None
    _memory : MemoryInterface = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, memory_db_type=MEMORY_MODE, **memory_instance_kwargs):
        initialize_pyrit(memory_db_type=memory_db_type, **memory_instance_kwargs)
        self._memory = CentralMemory.get_memory_instance()

    def get_memory(self) -> MemoryInterface:
        return self._memory

    def dump_to_json(self):
        self._memory.export_conversations(file_path=pathlib.Path(MEMORY_RESULTS_DIR)/"conversation.json")

    def dump_debug_log(self, *, log_name="debug.log", orchestrator_id=None):
        log_file = os.path.join(MEMORY_RESULTS_DIR, log_name)
        with open(log_file, "w", encoding="utf-8") as f:
            element = self._memory.get_prompt_request_pieces(orchestrator_id=orchestrator_id)
            for r in element:
                f.write("\n" + "*"*100 + "\n")
                f.write(str(r.to_dict()))

        '''
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

    def generate_csv_report(self, out_path="run_summary.csv"):
        '''
        out_file = os.path.join(RESULTS_DIR, out_path)

        # TODO find right variables to pick
        if MEMORY_MODE == DUCK_DB and DUCK_DB_PATH and os.path.exists(DUCK_DB_PATH):
            # Export directly from the DuckDB database
            con = duckdb.connect(DUCK_DB_PATH)
            df = con.execute("""
                SELECT 
                    rr.id, 
                    rr.prompt, 
                    rr.response, 
                    s.name AS score_name, 
                    s.value AS score_value
                FROM request_response rr
                LEFT JOIN score s ON rr.id = s.request_response_id
            """).fetchdf()
        else:
            # Fallback: build dataframe manually from memory
            rows = []
            for rr in self._memory.get_prompt_request_pieces():
                matched_scores = self._memory.get_scores_by_prompt_ids(rr.id)
                if not matched_scores:
                    rows.append({
                        "id": rr.id,
                        "prompt": rr.original_value,
                        "response": rr.original_value,
                        "score_name": None,
                        "score_value": None,
                    })
                else:
                    for s in matched_scores:
                        rows.append({
                            "id": rr.id,
                            "prompt": rr.original_value,
                            "response": rr.original_value,
                            "score_name": s.,
                            "score_value": s.value,
                        })
            df = pd.DataFrame(rows)

        df.to_csv(out_file, index=False)
        print(f"[✅] CSV report saved to {out_file}")
        '''
        pass
