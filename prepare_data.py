import argparse
import os

import torch

from dataset import CircleDataset


def prepare_data(args):
    """
    Prepares train, val and test datasets.
    Saves the prepared datasets.
    """

    # prepare datasets
    train_dataset = CircleDataset(
        n_samples=args.n_train,
        noise_level=args.noise_level,
        img_size=args.img_size,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )
    val_dataset = CircleDataset(
        n_samples=args.n_val,
        noise_level=args.noise_level,
        img_size=args.img_size,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )
    test_dataset = CircleDataset(
        n_samples=args.n_test,
        noise_level=args.noise_level,
        img_size=args.img_size,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )

    # create directory for saving datasets
    filepath = "./data"
    if args.filepath:
        filepath = args.filepath
    filepath = f"{filepath}_noise_level={args.noise_level}_img_size={args.img_size}"

    os.makedirs(filepath, exist_ok=True)

    # save the prepared datasets
    torch.save(train_dataset, f"{filepath}/train_dataset.pt")
    torch.save(val_dataset, f"{filepath}/val_dataset.pt")
    torch.save(test_dataset, f"{filepath}/test_dataset.pt")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--min_radius", type=float, default=None)
    parser.add_argument("--max_radius", type=float, default=None)
    parser.add_argument("--n_train", type=int, default=30000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--filepath", type=str, default=None)

    args = parser.parse_args()
    print(f"Preparing data with args: {vars(args)}")
    return args


if __name__ == "__main__":
    args = get_args()
    prepare_data(args)
