from pyrit.common import (
    initialize_pyrit,
    IN_MEMORY,
    DUCK_DB
)
from pyrit.memory import MemoryInterface
from pyrit.memory.central_memory import CentralMemory

import os

STORAGE_DIR = 'storage'
MEMORY_MODE = IN_MEMORY # os.getenv("MEMORY_MODE", "DUCK_DB").upper()
DEBUG_LOGGING = True # os.getenv("DEBUG_LOG", "false").lower() == "true"
DUCK_DB_PATH = os.path.join(STORAGE_DIR, "pyrit_results.db")

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

