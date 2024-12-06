import fire

from sweepr import Sweep, StatelessProvider
from sweepr.providers.wandb import WandBProvider  # noqa: F401
from sweepr.executors.python import PythonExecutor
from sweepr.executors.pueue import PueueExecutor
from sweepr.executors.slurm import SlurmExecutor


def main(out=None):
    sweep = (
        Sweep()
        .args(
            {
                "seed": 137,
                "batch_size": 8,
                "dataset": [
                    "gsm8k",
                    "math500:algebra",
                    "math500:counting_and_probability",
                    "math500:geometry",
                    "math500:intermediate_algebra",
                    "math500:number_theory",
                    "math500:prealgebra",
                    "math500:precalculus",
                ],
            }
        )
        .args(
            {
                "seed": 137,
                "dataset": ["rng"],
                "evaluator": ["rng", "srng"],
            }
        )
        .include(
            [
                ({"dataset": ["gsm8k", "^math500:"]}, {"evaluator": "math"}),
                ({"dataset": "^math500:"}, {"batch_size": 4}),
                ({"dataset": "rng"}, {"batch_size": 16}),
            ]
        )
        .exclude([{"dataset": "^math500:"}])
        .provider(StatelessProvider())
        # .provider(WandBProvider())
        ## Last one takes effect.
        .executor(PythonExecutor(file="test.py"))
        .executor(PueueExecutor(file="test.py", gpus=1))
        .executor(
            SlurmExecutor(file="test.py", account="sk", timelimit=12, gpus="a100:1")
        )
    )

    breakpoint()

    # sweep.write_json(file=out, indent=2, jsonl=True or False)
    sweep.write_bash(file=out)


if __name__ == "__main__":
    fire.Fire(main)
