from sweepr import Sweep
import fire


def main():
    sweep = (
        Sweep(executable=["python", "test.py"])
        .args(
            {
                "seed": 137,
                "batch_size": 8,
                "dataset": ["rng", "gsm8k", "math"],
            }
        )
        .include(
            [
                ({"dataset": ["gsm8k", "math"]}, {"evaluator": "math"}),
                ({"dataset": "math"}, {"batch_size": 4}),
            ]
        )
        .exclude({"batch_size": 4})
    )

    for config in sweep:
        print(config)


if __name__ == "__main__":
    fire.Fire(main)
