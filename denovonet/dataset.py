"""
dataset.py

Copyright (c) 2023 Karolis Sablauskas
Copyright (c) 2023 Gelana Khazeeva

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

import datetime
import glob
import multiprocessing as mp
import os
import sys
from functools import partial

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import pysam

# import tensorflow as tf
import torch

from denovonet.variants import SingleVariant, TrioVariant


class Dataset:
    """
    class Dataset helps to standartize variants for DeNovoCNN,
    saves variants as images in parallel (for training) and applies DeNovoCNN in parallel on variants list
    """

    def __init__(self, dataset=None, convert_to_inner_format=True):

        self.convert_to_inner = convert_to_inner_format
        self.dataset = dataset
        print("self.dataset", self.dataset)

        # represent variants in unified way
        self.standartize_variants()

    def standartize_variants(self):
        self.dataset[["Reference", "Variant"]] = (
            self.dataset[["Reference", "Variant"]].fillna("").astype(str)
        )

        self.dataset["Variant"] = self.dataset["Variant"].apply(
            lambda x: str(x).split(",")
        )
        self.dataset = self.dataset.explode("Variant")
        self.dataset["Variant"] = self.dataset["Variant"].apply(str.strip)

        self.dataset["Variant type"] = self.dataset.apply(get_variant_class, axis=1)

        # dummy for Target column
        if "Target" not in self.dataset.columns:
            self.dataset["Target"] = ""

        self.dataset = self.dataset.apply(remove_matching_nucleotides, axis=1)

        if self.convert_to_inner:
            self.dataset.loc[
                self.dataset["Variant type"] == "Insertion", "Start position std"
            ] = (
                self.dataset.loc[
                    self.dataset["Variant type"] == "Insertion", "Start position std"
                ]
                - 1
            )

            self.dataset.loc[
                self.dataset["Variant type"] == "Insertion", "End position std"
            ] = (
                self.dataset.loc[
                    self.dataset["Variant type"] == "Insertion", "End position std"
                ]
                - 1
            )

    def save_images(self, folder, reference_genome_path, n_jobs=-1):
        self.dataset["Key"] = (
            self.dataset["Child"].astype("str")
            + "_"
            + self.dataset["Chromosome"].astype("str")
            + "_"
            + self.dataset["Start position"].astype("str")
            + "_"
            + self.dataset["Reference"].astype("str")
            + "_"
            + self.dataset["Variant"].astype("str")
        )

        # create subdirectories
        for target_value in self.dataset["Target"].unique().tolist():
            dir_name = os.path.join(folder, str(target_value))

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        # save images in parallel
        print("\nStart saving images to subdirectories in", folder)

        if n_jobs == -1:
            # use all CPUs
            n_jobs = mp.cpu_count()

        print(f"Using {n_jobs} CPUs")
        sys.stdout.flush()
        start = datetime.datetime.now()

        pool = mp.Pool(n_jobs)

        _ = pool.map(
            partial(
                save_image, folder=folder, reference_genome_path=reference_genome_path
            ),
            self.dataset[
                [
                    "Chromosome",
                    "Start position std",
                    "End position std",
                    "Variant type",
                    "Child BAM",
                    "Father BAM",
                    "Mother BAM",
                    "Key",
                    "Target",
                ]
            ].values,
        )

        pool.close()
        print("Saving images finished, time elapsed:", datetime.datetime.now() - start)
        sys.stdout.flush()

    def apply_model(
        self, models_cfg, reference_genome_path, n_jobs=-1, batch_size=1000
    ):

        # apply models in parallel
        print("\nStart applying DeNovoCNN")

        if n_jobs == -1:
            # use all CPUs
            n_jobs = mp.cpu_count()

        dataset_batches = self.dataset[
            [
                "Chromosome",
                "Start position std",
                "End position std",
                "Variant type",
                "Child BAM",
                "Father BAM",
                "Mother BAM",
            ]
        ].values.tolist()

        dataset_batches = [
            dataset_batches[x : x + batch_size]
            for x in range(0, len(dataset_batches), batch_size)
        ]

        print(f"Using {n_jobs} CPUs")
        sys.stdout.flush()
        start = datetime.datetime.now()

        pool = mp.Pool(n_jobs)

        results = pool.map(
            partial(
                apply_model_batch,
                models_cfg=models_cfg,
                reference_genome_path=reference_genome_path,
            ),
            dataset_batches,
        )

        pool.close()
        print(
            "Applying DeNovoCNN finished, time elapsed:",
            datetime.datetime.now() - start,
        )
        sys.stdout.flush()

        # flattern results
        results = [pred for sub_pred in results for pred in sub_pred]

        self.dataset["DeNovoCNN probability"] = [res[0] for res in results]
        self.dataset["Child coverage"] = [res[1][0] for res in results]
        self.dataset["Father coverage"] = [res[1][1] for res in results]
        self.dataset["Mother coverage"] = [res[1][2] for res in results]

    def apply_model_onnx(
        self, models_cfg, reference_genome_path, n_jobs=-1, batch_size=1000
    ):

        # apply models in parallel
        print("\nStart applying DeNovoCNN")

        if n_jobs == -1:
            # use all CPUs
            n_jobs = mp.cpu_count()

        dataset_batches = self.dataset[
            [
                "Chromosome",
                "Start position std",
                "End position std",
                "Variant type",
                "Child BAM",
                "Father BAM",
                "Mother BAM",
            ]
        ].values.tolist()

        dataset_batches = [
            dataset_batches[x : x + batch_size]
            for x in range(0, len(dataset_batches), batch_size)
        ]

        print(f"Using {n_jobs} CPUs")
        sys.stdout.flush()
        start = datetime.datetime.now()

        pool = mp.Pool(n_jobs)

        results = pool.map(
            partial(
                apply_model_batch_onnx,
                models_cfg=models_cfg,
                reference_genome_path=reference_genome_path,
            ),
            dataset_batches,
        )

        pool.close()
        print(
            "Applying DeNovoCNN finished, time elapsed:",
            datetime.datetime.now() - start,
        )
        sys.stdout.flush()

        # flattern results
        results = [pred for sub_pred in results for pred in sub_pred]

        self.dataset["DeNovoCNN probability"] = [res[0] for res in results]
        self.dataset["Child coverage"] = [res[1][0] for res in results]
        self.dataset["Father coverage"] = [res[1][1] for res in results]
        self.dataset["Mother coverage"] = [res[1][2] for res in results]

    def save_dataset(self, output_path, output_denovocnn_format=False):
        if output_denovocnn_format == "true":
            output_columns = [
                "Chromosome",
                "Start position std",
                "End position std",
                "Reference std",
                "Variant std",
                "DeNovoCNN probability",
                "Child coverage",
                "Father coverage",
                "Mother coverage",
                "Child BAM",
                "Father BAM",
                "Mother BAM",
            ]
        else:
            output_columns = [
                "Chromosome",
                "Start position",
                "Reference",
                "Variant",
                "DeNovoCNN probability",
                "Child coverage",
                "Father coverage",
                "Mother coverage",
                "Child BAM",
                "Father BAM",
                "Mother BAM",
            ]

        self.dataset[output_columns].to_csv(output_path, sep="\t", index=False)


def get_variant_class(row):
    """
    Defining variant type (SNP, insertion, deletion)
    based on reference and alternative alleles

    Parameters:
    row: DataFrame row

    Returns:
    Substitution, Deletion, Insertion or Unknown
    """
    reference, variant = row["Reference"], row["Variant"]

    if len(reference) == len(variant):
        return "Substitution"
    elif len(reference) > len(variant):
        return "Deletion"
    elif len(reference) < len(variant):
        return "Insertion"

    return "Unknown"


def generate_images_from_folder(
    folder, save_folder, reference_genome_path, convert_to_inner_format, n_jobs=-1
):
    files = glob.glob(f"{folder}/*_*.csv")

    for dataset_path in files:
        print("\nLoading images for", dataset_path)
        dataset_type, variant_type = tuple(
            dataset_path.split("/")[-1].split(".")[0].split("_")
        )

        full_save_folder = os.path.join(save_folder, variant_type, dataset_type)

        dataset = Dataset(
            dataset=pd.read_csv(dataset_path, sep="\t"),
            convert_to_inner_format=convert_to_inner_format,
        )

        dataset.save_images(
            folder=full_save_folder,
            reference_genome_path=reference_genome_path,
            n_jobs=n_jobs,
        )


def remove_matching_nucleotides(row):
    """
    removes matching nucleotides in Reference and Variant from beginning and end,
    updates variant positions accordingly,
    saves the results in new columns
    :param row: row from a processed dataframe
    """
    row["Reference std"] = row["Reference"]
    row["Variant std"] = row["Variant"]
    row["Start position std"] = row["Start position"]

    # remove from the end
    if len(row["Reference std"]) == len(row["Variant std"]):
        end_prefix = os.path.commonprefix(
            [row["Reference std"][::-1], row["Variant std"][::-1]]
        )

        row["Reference std"] = row["Reference std"][
            : (len(row["Reference std"]) - len(end_prefix))
        ]
        row["Variant std"] = row["Variant std"][
            : (len(row["Variant std"]) - len(end_prefix))
        ]

    # remove from the beginning
    prefix = os.path.commonprefix([row["Reference std"], row["Variant std"]])

    row["Start position std"] += len(prefix)
    row["Reference std"] = row["Reference std"][len(prefix) :]
    row["Variant std"] = row["Variant std"][len(prefix) :]

    row["End position std"] = row["Start position std"] + max(
        len(row["Reference std"]), 1
    )

    return row


def load_variant(chromosome, start, end, bam, reference_genome):
    """
    SingleVariant class initialization wrapper
    :param chromosome: chromosome
    :param start: variant start
    :param end: variant end
    :param bam: corresponding bam.cram file
    :param reference_genome: reference genome AligmnentFile
    :return: SingleVariant class
    """

    try:
        return SingleVariant(
            str(chromosome), int(start), int(end), bam, reference_genome
        )
    except (ValueError, KeyError):
        if chromosome[:3] != "chr":
            chromosome = "chr" + chromosome
            return SingleVariant(
                str(chromosome), int(start), int(end), bam, reference_genome
            )
        elif chromosome[:3] == "chr":
            chromosome = chromosome[:3]
            return SingleVariant(
                str(chromosome), int(start), int(end), bam, reference_genome
            )
        else:
            raise


def get_image(row, reference_genome_path):
    """
    creates TrioVariant class object for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param reference_genome_path: path to a reference genome
    """
    print("row", row)
    chromosome, start, end, _, child_bam, father_bam, mother_bam, key, target = tuple(
        row
    )

    # create image
    reference_genome = pysam.FastaFile(reference_genome_path)
    child_variant = load_variant(
        str(chromosome), int(start), int(end), child_bam, reference_genome
    )
    father_variant = load_variant(
        str(chromosome), int(start), int(end), father_bam, reference_genome
    )
    mother_variant = load_variant(
        str(chromosome), int(start), int(end), mother_bam, reference_genome
    )
    trio_variant = TrioVariant(child_variant, father_variant, mother_variant)

    return trio_variant


def save_image(row, folder, reference_genome_path):
    """
    saves RGB image for DeNovoCNN for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param folder: folder to place an image
    :param reference_genome_path: path to a reference genome
    """
    (
        chromosome,
        start,
        end,
        var_type,
        child_bam,
        father_bam,
        mother_bam,
        key,
        target,
    ) = tuple(row)

    trio_variant = get_image(row, reference_genome_path)

    # swap dimensions
    swapped_image = np.zeros(trio_variant.image.shape)
    swapped_image[:, :, 0], swapped_image[:, :, 1], swapped_image[:, :, 2] = (
        trio_variant.image[:, :, 2],
        trio_variant.image[:, :, 1],
        trio_variant.image[:, :, 0],
    )

    # save image
    output_path = os.path.join(folder, target, f"{key}.png")
    cv2.imwrite(output_path, swapped_image)

    return output_path


def load_models(models_cfg):
    """
    loads Substitution, Deletion and Insertion models from paths specified in models_cfg
    """
    models_dict = {
        "Substitution": tf.keras.models.load_model(
            models_cfg["snp_model"], compile=False
        ),
        "Deletion": tf.keras.models.load_model(models_cfg["del_model"], compile=False),
        "Insertion": tf.keras.models.load_model(models_cfg["ins_model"], compile=False),
    }

    return models_dict


def load_models_onnx(models_cfg):
    """
    loads Substitution, Deletion and Insertion models from paths specified in models_cfg
    """
    providers = (
        "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    )
    snp_session = ort.InferenceSession(models_cfg["snp_model"], providers=[providers])
    insertion_session = ort.InferenceSession(
        models_cfg["ins_model"], providers=[providers]
    )
    deletion_session = ort.InferenceSession(
        models_cfg["del_model"], providers=[providers]
    )
    models_dict = {
        "Substitution": snp_session,
        "Deletion": deletion_session,
        "Insertion": insertion_session,
    }

    return models_dict


def apply_model(row, models_dict, reference_genome_path):
    """
    applies DeNovoCNN models from models_dict to a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param models_dict: contains information about DeNovoCNN models paths
    :param reference_genome_path: path to a reference genome
    """
    _, _, _, var_type, _, _, _ = tuple(row)

    trio_variant = get_image(tuple(row) + (None, None), reference_genome_path)

    try:
        # predict
        prediction = trio_variant.predict(models_dict[var_type])
        prediction_dnm = str(round(1.0 - prediction[0, 0], 3))

        child_coverage = trio_variant.child_variant.start_coverage
        father_coverage = trio_variant.father_variant.start_coverage
        mother_coverage = trio_variant.mother_variant.start_coverage
    except KeyError:
        print("Failed in:")
        print("\t", row)
        raise

    return prediction_dnm, (child_coverage, father_coverage, mother_coverage)


def apply_model_onnx(row, models_dict, reference_genome_path):
    """
    applies DeNovoCNN models from models_dict to a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param models_dict: contains information about DeNovoCNN models paths
    :param reference_genome_path: path to a reference genome
    """
    _, _, _, var_type, _, _, _ = tuple(row)

    trio_variant = get_image(tuple(row) + (None, None), reference_genome_path)

    try:
        # predict
        prediction = trio_variant.predict_onnx(models_dict[var_type])
        prediction_dnm = str(round(1.0 - prediction[0, 1], 3))

        child_coverage = trio_variant.child_variant.start_coverage
        father_coverage = trio_variant.father_variant.start_coverage
        mother_coverage = trio_variant.mother_variant.start_coverage
    except KeyError:
        print("Failed in:")
        print("\t", row)
        raise

    return prediction_dnm, (child_coverage, father_coverage, mother_coverage)


def apply_model_batch(rows, models_cfg, reference_genome_path):
    """
    applies  DeNovoCNN models from models_cfg to a batch of rows from a processed dataframe
    """
    models_dict = load_models(models_cfg)
    return [apply_model(row, models_dict, reference_genome_path) for row in rows]


def apply_model_batch_onnx(rows, models_cfg, reference_genome_path):
    """
    applies  DeNovoCNN models from models_cfg to a batch of rows from a processed dataframe
    """
    models_dict = load_models_onnx(models_cfg)
    print(models_dict)
    return [apply_model_onnx(row, models_dict, reference_genome_path) for row in rows]


def apply_models_on_trio(
    variants_list,
    output_path,
    child_bam,
    father_bam,
    mother_bam,
    snp_model,
    del_model,
    ins_model,
    ref_genome,
    output_denovocnn_format,
    convert_to_inner_format,
    n_jobs,
):
    """
    applies DeNovoCNN models in parallel to all variants specified in variants_list for a trio
    """

    trio_cfg = {
        "Child": {"bam_path": child_bam},
        "Father": {"bam_path": father_bam},
        "Mother": {"bam_path": mother_bam},
    }

    models_cfg = {
        "snp_model": snp_model,
        "del_model": del_model,
        "ins_model": ins_model,
    }

    # read variants list
    variant_cols = ["Chromosome", "Start position", "Reference", "Variant", "extra"]

    dataset = pd.read_csv(variants_list, sep="\t", names=variant_cols)

    # fill in sample information
    for sample in trio_cfg:
        dataset[sample] = trio_cfg[sample].get("id", "")
        dataset[f"{sample} BAM"] = trio_cfg[sample]["bam_path"]

    print(f"Start apply in parallel, n_jobs={n_jobs}")

    # apply models
    dataset = Dataset(dataset=dataset, convert_to_inner_format=convert_to_inner_format)

    dataset.apply_model(
        models_cfg=models_cfg,
        reference_genome_path=ref_genome,
        n_jobs=n_jobs,
        batch_size=1000,
    )

    dataset.save_dataset(
        output_path=output_path, output_denovocnn_format=output_denovocnn_format
    )


def apply_models_on_trio_onnx(
    variants_list,
    output_path,
    child_bam,
    father_bam,
    mother_bam,
    snp_model,
    del_model,
    ins_model,
    ref_genome,
    output_denovocnn_format,
    convert_to_inner_format,
    n_jobs,
):
    """
    applies DeNovoCNN models in parallel to all variants specified in variants_list for a trio
    """
    trio_cfg = {
        "Child": {"bam_path": child_bam},
        "Father": {"bam_path": father_bam},
        "Mother": {"bam_path": mother_bam},
    }

    models_cfg = {
        "snp_model": snp_model,
        "del_model": del_model,
        "ins_model": ins_model,
    }

    # # read variants list
    variant_cols = ["Chromosome", "Start position", "Reference", "Variant", "extra"]

    dataset = pd.read_csv(variants_list, sep="\t", names=variant_cols)

    # fill in sample information
    for sample in trio_cfg:
        dataset[sample] = trio_cfg[sample].get("id", "")
        dataset[f"{sample} BAM"] = trio_cfg[sample]["bam_path"]

    print(f"Start apply in parallel, n_jobs={n_jobs}")

    # apply models
    dataset = Dataset(dataset=dataset, convert_to_inner_format=convert_to_inner_format)

    dataset.apply_model_onnx(
        models_cfg=models_cfg,
        reference_genome_path=ref_genome,
        n_jobs=n_jobs,
        batch_size=1000,
    )

    dataset.save_dataset(
        output_path=output_path, output_denovocnn_format=output_denovocnn_format
    )
