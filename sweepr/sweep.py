from tqdm.auto import tqdm
import polars as pl
from functools import reduce
from operator import iand, ior

from .types import Executable, ArgsMatrix, Includes, Excludes
from .utils import iter_dict


class Sweep:
    def __init__(self, executable: Executable):
        if isinstance(executable, str):
            executable = executable.split(" ")

        self.executable: list = executable

        self.df: pl.DataFrame = None

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        yield from self.df.iter_rows(named=True)

    def args(self, matrix: ArgsMatrix):
        all_args = iter_dict(matrix)

        if self.df is None:
            self.df = pl.DataFrame(all_args)
        else:
            raise NotImplementedError

        self.df = self.df.unique()

        return self

    def _check_cols_exist(self, exist_cols, add_missing=False):
        exist_cols = set(exist_cols)
        cols = set(self.df.columns)

        missing_cols = exist_cols - cols
        if missing_cols:
            if add_missing:
                self.df = self.df.with_columns(
                    **{k: pl.lit(None) for k in missing_cols}
                )
            else:
                msg = ", ".join([f'"{c}"' for c in missing_cols])
                raise ValueError(f"{len(missing_cols)} column(s) missing: {msg}.")

    def _prepare_match_conditions(self, match_dict: ArgsMatrix):
        ## TODO: replace with custom match function supporting regex for strings.
        return reduce(
            ior,
            [
                reduce(iand, [pl.col(k) == v for k, v in md.items()])
                for md in iter_dict(match_dict)
            ],
        )

    def include(self, includes: Includes):
        assert self.df is not None, "Did you set .args(...) first?"

        if not isinstance(includes, list):
            includes = [includes]

        for match_dict, include_dict in tqdm(includes, leave=False):
            self._check_cols_exist(match_dict.keys())

            self._check_cols_exist(include_dict.keys(), add_missing=True)

            self.df = self.df.with_columns(
                pl.when(self._prepare_match_conditions(match_dict))
                .then(pl.struct(**{k: pl.lit(v) for k, v in include_dict.items()}))
                .otherwise(pl.struct(*include_dict.keys()))
                .struct.unnest()
            )

        self.df = self.df.unique()

        return self

    def exclude(self, excludes: Excludes):
        assert self.df is not None, "Did you set .args(...) first?"

        if not isinstance(excludes, list):
            excludes = [excludes]

        for match_dict in tqdm(excludes, leave=False):
            self._check_cols_exist(match_dict.keys())

            self.df = self.df.filter(~self._prepare_match_conditions(match_dict))

        return self
