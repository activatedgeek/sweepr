from enum import Enum


class RunEnv(str, Enum):
    HASH = "RUN_HASH"
    TAGS = "RUN_TAGS"
