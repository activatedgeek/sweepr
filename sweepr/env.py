from enum import Enum


class RunEnv(str, Enum):
    HASH = "RUN_HASH"
    TAGS = "RUN_TAGS"


class PueueEnv(str, Enum):
    GPUS = "GPUS"


class SlurmEnv(str, Enum):
    TIMELIMIT = "HH"
    GPUS = "GPUS"
    ACCOUNT = "SBATCH_ACCOUNT"
