import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CircleDetector
from prepare_data import prepare_data
from utils import accuracy


def load_model(filename):
    """Loads saved torch model"""
    return torch.load(filename)


def save_model(model, filename):
    """Saves torch model"""
    torch.save(model.state_dict(), filename)


def model_eval(dataloader, model, device, threshold):
    """
    Evaluate the model.
    Calculates accuracy between predictions and actuals
    """
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval", disable=True)):
        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets

        outputs = model(inputs)
        outputs = outputs.detach().cpu().numpy()

        y_true.extend(targets)
        y_pred.extend(outputs)

    return accuracy(y_true, y_pred, threshold)


def run(args):
    """Runs training and testing of the model."""

    # check for gpu availablility. If available, use it.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    filepath = "./data"
    if args.filepath:
        filepath = args.filepath
    filepath = f"{filepath}_noise_level={args.noise_level}_img_size={args.img_size}"

    # check if data is prepared or not
    if not os.path.exists(filepath):
        print("Data not found!. Preparing new data.")
        prepare_data(args)

    # load datasets
    train_dataset = torch.load(f"{filepath}/train_dataset.pt")
    val_dataset = torch.load(f"{filepath}/val_dataset.pt")
    test_dataset = torch.load(f"{filepath}/test_dataset.pt")

    # create directory for saving model
    savepath = "./saved"
    if args.savepath:
        savepath = args.savepath

    os.makedirs(savepath, exist_ok=True)
    filename = f"{savepath}/CircleDetector_noise_level={args.noise_level}_img_size={args.img_size}_threshold={args.threshold}.pt"

    # prepare dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # initialize CircleDetector model
    model = CircleDetector(img_size=args.img_size).to(device)
    print(model)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of parameters:", params)

    # specify the loss and the optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0
    best_epoch = -1

    # run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            optimizer.zero_grad()

            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_accuracy = model_eval(train_dataloader, model, device, args.threshold)
        val_accuracy = model_eval(val_dataloader, model, device, args.threshold)
        print(
            f"epoch {epoch}: train loss :: {train_loss :.3f}, train_accuracy acc :: {train_accuracy :.3f}, dev acc :: {val_accuracy :.3f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            # save if current accuracy is best
            save_model(model, filename)
        elif epoch - best_epoch > args.early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break

    print(f"Best model at epoch {best_epoch} with val accuracy {best_accuracy}")

    # load the best model
    model = load_model(filename)

    # evaluate test data
    test_accuracy = model_eval(test_dataloader, model, device, args.threshold)
    print(f"Test accuracy is {test_accuracy}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--early_stop_thresh", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    print(f"Running model training and testing with args: {vars(args)}")
    return args


if __name__ == "__main__":
    args = get_args()
    run(args)
