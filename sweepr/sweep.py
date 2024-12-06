from typing import Iterator, Optional, List, Union, TextIO
import sys
from contextlib import nullcontext
from tqdm.auto import tqdm
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
import uuid
import polars as pl
from functools import reduce
from operator import iand, ior

from .types import Program, Arg, ArgsDict, EnvDict, ArgsMatrix, Includes, Excludes
from .utils import iter_dict


@dataclass
class Run:
    program: List[str]
    args: Optional[ArgsDict] = field(default_factory=lambda: {})
    env: Optional[EnvDict] = field(default_factory=lambda: {})

    @property
    def argv(self):
        return (
            [f"RUN_HASH={self.hash}"]
            + [f"{k}={v}" for k, v in self.env.items()]
            + self.program
            + [f"--{k}={v}" for k, v in self.args.items()]
        )

    def __str__(self):
        return " ".join(self.argv)

    def __hash__(self):
        return self.hash

    @property
    def hash(self):
        dict_str = (
            str({k: getattr(self, k) for k in self.fields})
            .encode("utf-8")
            .decode("utf-8")
        )

        return str(uuid.uuid5(uuid.NAMESPACE_DNS, dict_str))[:8]

    @property
    def fields(self):
        return [f.name for f in dataclasses.fields(Run) if not f.name.startswith("_")]


class Sweep:
    def __init__(self, program: Program):
        if isinstance(program, str):
            program = program.split(" ")

        self._program: list = program

        self._df: pl.DataFrame = None

    def __len__(self):
        return len(self._df)

    def __iter__(self) -> Iterator[Run]:
        for row in self._df.iter_rows(named=True):
            run = Run(
                program=self._program,
                args={k: v for k, v in row.items() if v is not None},
            )

            yield run

    def args(self, matrix: ArgsMatrix):
        all_args = iter_dict(matrix)

        new_df = pl.DataFrame(all_args)

        if self._df is None:
            self._df = new_df
        else:
            self._check_cols_exist(new_df.columns, add_missing=True)
            self._check_cols_exist(self._df.columns, add_missing=True, df=new_df)

            self._df = pl.concat([self._df, new_df], how="align", rechunk=True)

        self._df = self._df.unique()

        return self

    def include(self, includes: Includes):
        assert self._df is not None, "Did you set .args(...) first?"

        if not isinstance(includes, list):
            includes = [includes]

        for match_dict, include_dict in tqdm(includes, leave=False):
            self._check_cols_exist(match_dict.keys())

            self._check_cols_exist(include_dict.keys(), add_missing=True)

            self._df = self._df.with_columns(
                pl.when(self._prepare_match_conditions(match_dict))
                .then(pl.struct(**{k: pl.lit(v) for k, v in include_dict.items()}))
                .otherwise(pl.struct(*include_dict.keys()))
                .struct.unnest()
            )

        self._df = self._df.unique()

        return self

    def exclude(self, excludes: Excludes):
        assert self._df is not None, "Did you set .args(...) first?"

        if not isinstance(excludes, list):
            excludes = [excludes]

        for match_dict in tqdm(excludes, leave=False):
            self._check_cols_exist(match_dict.keys())

            self._df = self._df.filter(~self._prepare_match_conditions(match_dict))

        return self

    def write_bash(self, file: Union[str, Path, TextIO] = None, delay: int = 3):
        with (
            open(file, "w")
            if isinstance(file, (str, Path))
            else nullcontext(file or sys.stdout) as file
        ):
            print("#!/usr/bin/env -S bash -l", file=file)
            print(file=file)

            for run in self:
                print(str(run), file=file)
                print(f"sleep $(( RANDOM % {delay} ))", file=file)
                print(file=file)

        return self

    def _check_cols_exist(self, exist_cols, add_missing=False, df=None):
        if df is None:
            df = self._df

        exist_cols = set(exist_cols)
        cols = set(df.columns)

        missing_cols = exist_cols - cols
        if missing_cols:
            if add_missing:
                df = df.with_columns(**{k: pl.lit(None) for k in missing_cols})
            else:
                msg = ", ".join([f'"{c}"' for c in missing_cols])
                raise ValueError(f"{len(missing_cols)} column(s) missing: {msg}.")

    def _prepare_match_conditions(self, match_dict: ArgsMatrix):
        def _expr(k: str, v: Arg):
            if isinstance(v, str):
                ## NOTE: Regex syntax at https://docs.rs/regex/latest/regex/#syntax
                return pl.col(k).str.contains(v, literal=False)
            return pl.col(k) == v

        return reduce(
            ior,
            [
                reduce(iand, [_expr(k, v) for k, v in md.items()])
                for md in iter_dict(match_dict)
            ],
        )
