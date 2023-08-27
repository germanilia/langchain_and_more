from enum import Enum, auto

class MemoryType(Enum):
    SUMMARY = auto()
    CONVERSATION = auto()

class ModelType(Enum):
    ANTHROPIC = auto()
    AZURE = auto()
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"

class ChainType(Enum):
    STUFF="stuff"
    MAP_REDUCE="map_reduce"
    REFINE="refine"