# call shadows and circuits to generate dataset

import torch


def generate_dataset_single_molecule(molecule, samples, max_time=1, time_steps=0.1):
    ...

    krdms = 0  # make it a Dataset object

    return krdms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate quantum shadow dataset for H2O molecule."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="h2o_shadows_dataset.pt",
        help="Path to save the generated dataset.",
    )

    args = parser.parse_args()
    path = args.path

    # make H2O dataset
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.7586, 0.5859, 0.0],
        [-0.7586, 0.5859, 0.0],
    ]

    dataset = generate_dataset_single_molecule(
        molecule=("H2O", symbols, coords),
        samples=1000,
        max_time=1.0,
        time_steps=0.1,
    )
    # save dataset
    torch.save(dataset, path)
