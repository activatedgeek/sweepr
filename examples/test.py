from sweepr import Sweep
import fire


def main(out=None):
    (
        Sweep(program=["python", "test.py"])
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
    ).write_bash(file=out)


if __name__ == "__main__":
    fire.Fire(main)
