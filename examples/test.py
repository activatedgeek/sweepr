from sweepr import Sweep
import fire


def main():
    (
        Sweep(program=["python", "test.py"])
        .args(
            {
                "seed": 137,
                "batch_size": 8,
                "dataset": ["gsm8k", "math"],
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
                ({"dataset": ["gsm8k", "math"]}, {"evaluator": "math"}),
                ({"dataset": "math"}, {"batch_size": 4}),
                ({"dataset": "rng"}, {"batch_size": 16}),
            ]
        )
    ).write_bash()


if __name__ == "__main__":
    fire.Fire(main)
