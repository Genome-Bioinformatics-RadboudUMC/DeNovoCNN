"""
training.py

Copyright (c) 2022 Karolis Sablauskas
Copyright (c) 2022 Gelana Khazeeva

This file is part of DeNovoCNN.

DeNovoCNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeNovoCNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import os

import onnxruntime as ort
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)
from torch import nn
from tqdm import tqdm
from denovonet.training.utils import (
    ImbalancedDatasetSampler,
    export_onnx_model,
    ort_inference,
)

# parse arguments
parser = argparse.ArgumentParser(description="Use DeNovoCNN.")

parser.add_argument(
    "--images",
    type=str,
    required=True,
    help='Path to the training images / val images, e.g "training_dataset/publish_images/insertion".',
)
parser.add_argument(
    "--workdir",
    type=str,
    required=True,
    help='Path to working directory, exporting model".',
)
parser.add_argument("--outname", type=str, required=True, help='Name of the model".')
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Batch size for training and inference. Default 32.",
)
parser.add_argument(
    "--epochs", default=7, type=int, help="NUmber of epochs for training. Default 7."
)


def training_pipeline(
    batch_size: int, images_folder: str, epochs: int, workdir: str, outname: str
) -> None:

    # Additional transforms, e.g. normalization can be added in the future
    transform = transforms.Compose([transforms.ToTensor()])
    path_to_train_images = os.path.join(images_folder, "train")

    trainset = torchvision.datasets.ImageFolder(
        root=path_to_train_images, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(trainset),
        num_workers=4,
    )

    path_to_val_images = os.path.join(images_folder, "val")
    valset = torchvision.datasets.ImageFolder(
        root=path_to_val_images, transform=transform
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=4
    )

    path_to_test_images = os.path.join(images_folder, "test")
    testset = torchvision.datasets.ImageFolder(
        root=path_to_test_images, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=4
    )

    model = torchvision.models.resnet18(num_classes=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        mean_train_loss = train_loss / len(trainset)

        y_true = []
        y_pred = []
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for (inputs, labels) in tqdm(valloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                y_true.append(labels.detach().cpu()[0])
                y_pred.append(predicted.detach().cpu().numpy()[0])
                val_loss += loss.item() * batch_size

        mean_val_loss = val_loss / len(valset)

        recall = round(recall_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        acc = round(accuracy_score(y_true, y_pred), 4)
        print(
            f"Epoch: {epoch} - mean_train_loss={mean_train_loss} mean_val_loss={mean_val_loss} recall={recall} precision={precision} accuracy={acc}"
        )

    print("Finished Training")

    # Export model
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    output_path = os.path.join(workdir, f"{outname}.onnx")
    export_onnx_model(model, 3, 160, 164, output_path)
    print(f"Final model exported at {output_path}")

    providers = (
        "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    )
    session = ort.InferenceSession(output_path, providers=[providers])

    y_true = []
    y_pred = []

    for data in tqdm(testloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = ort_inference(session, inputs)
        _, predicted = torch.max(outputs.data, 1)

        y_true.append(labels.detach().cpu()[0])
        y_pred.append(predicted.detach().cpu().numpy()[0])

    recall = round(recall_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred), 4)
    acc = round(accuracy_score(y_true, y_pred), 4)
    print(
        f"Perfomance on the test dataset using ONNX model: recall={recall} precision={precision} accuracy={acc}"
    )


if __name__ == "main":
    args = parser.parse_args()
    training_pipeline(
        batch_size=args.batch_size,
        images_folder=args.images,
        epochs=args.epochs,
        workdir=args.workdir,
        outname=args.outname,
    )
