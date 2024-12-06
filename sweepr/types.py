from typing import Dict, List, Union, Tuple, TypeAlias, TypedDict, Optional


Program: TypeAlias = Union[str, List[str]]

Arg: TypeAlias = Union[str, int, float]

ArgsDict: TypeAlias = Dict[str, Arg]

EnvDict: TypeAlias = Dict[str, str]


class PueueConfigDict(TypedDict):
    gpus: int


class SlurmConfigDict(TypedDict):
    gpus: Optional[Union[int, str]]
    timelimit: Optional[int]
    account: Optional[str]


ArgsMatrix: TypeAlias = Dict[str, Union[Arg, List[Arg]]]

IncludeTuple: TypeAlias = Tuple[ArgsMatrix, ArgsDict]

Includes: TypeAlias = Union[IncludeTuple, List[IncludeTuple]]

Excludes: TypeAlias = Union[ArgsMatrix, List[ArgsMatrix]]
