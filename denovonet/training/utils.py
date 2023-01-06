"""
utils.py

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

import glob
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils.data.sampler import Sampler


class ImbalancedDatasetSampler(Sampler):
    """
    From https://github.com/ufoym/imbalanced-dataset-sampler

    Samples elements randomly from a given list of
        indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def export_onnx_model(model, image_channels, image_height, image_width, output_path):
    x = torch.randn(
        1,
        image_channels,
        image_height,
        image_width,
        requires_grad=True,
    )

    bs = "batch_size"
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: bs},  # variable length axes
        "output": {0: bs},
    }
    torch.onnx.export(
        model,  # model being run
        x.cuda(),  # model input (or a tuple for multiple inputs)
        output_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=output_names,  # the model's output names
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )


def ort_inference(session, tensor):
    io_binding = session.io_binding()
    io_binding.bind_cpu_input(session.get_inputs()[0].name, tensor.cpu().numpy())
    io_binding.bind_output("output")
    session.run_with_iobinding(io_binding)
    predictions = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0])
    return predictions


def predict_image_path_onxx(image_path: str, model) -> np.array:
    """
    Applies the model to RGB image within the image_path
    and gets DNM prediction
    """
    import torch
    from torchvision import transforms

    image = np.array(Image.open(image_path)) / 255.0

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).type(torch.FloatTensor)
    input_tensor = input_tensor[None]  # expand dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor.to(device)
    prediction = ort_inference(model, input_tensor)
    softmax = torch.nn.Softmax(dim=1)
    prediction = softmax(prediction)
    prediction = prediction.detach().cpu().numpy()

    return prediction


def apply_model_image_onxx(image_path, model):
    return 1.0 - predict_image_path_onxx(image_path, model=model)[0, 1]


def apply_model_images_onxx(model_path, images_folder):
    import onnxruntime as ort
    import torch

    providers = (
        "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    )

    model = ort.InferenceSession(model_path, providers=[providers])

    images = glob.glob(f"{images_folder}/*/*.png")

    predictions = []

    for image_path in tqdm.tqdm(images):
        predictions.append(apply_model_image_onxx(image_path, model=model))

    target = [1 if "DNM" in path else 0 for path in images]

    return images, target, predictions
