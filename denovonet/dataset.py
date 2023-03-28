'''
dataset.py

Copyright (c) 2021 Karolis Sablauskas
Copyright (c) 2021 Gelana Khazeeva

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
'''

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime
from functools import partial
import glob
import cv2
import pysam
import tensorflow as tf
from denovonet.image_generation import gen_trio_img


class Dataset:
    """
    class Dataset helps to standartize variants for DeNovoCNN,
    saves variants as images in parallel (for training) and applies DeNovoCNN in parallel on variants list
    """
    def __init__(self, dataset=None, convert_to_inner_format=True):

        self.convert_to_inner = convert_to_inner_format
        self.dataset = dataset

        # represent variants in unified way
        self.standartize_variants()

    def standartize_variants(self):
        self.dataset[['Reference', 'Variant']] = self.dataset[
            ['Reference', 'Variant']].fillna('').astype(str)

        self.dataset['Variant'] = self.dataset['Variant'].apply(lambda x: str(x).split(','))
        self.dataset = self.dataset.explode('Variant')
        self.dataset['Variant'] = self.dataset['Variant'].apply(str.strip)

        self.dataset['Variant type'] = self.dataset.apply(get_variant_class, axis=1)

        # dummy for Target column
        if 'Target' not in self.dataset.columns:
            self.dataset['Target'] = ''

        self.dataset = self.dataset.apply(remove_matching_nucleotides, axis=1)

        if self.convert_to_inner:
            self.dataset.loc[self.dataset['Variant type'] == 'Insertion', 'Start position std'] = (
                self.dataset.loc[self.dataset['Variant type'] == 'Insertion', 'Start position std'] - 1
            )

            self.dataset.loc[self.dataset['Variant type'] == 'Insertion', 'End position std'] = (
                self.dataset.loc[self.dataset['Variant type'] == 'Insertion', 'End position std'] - 1
            )

    def save_images(self, folder, reference_genome_path, n_jobs=-1):
        self.dataset['Key'] = (
                self.dataset['Child'].astype('str') + "_" +
                self.dataset['Chromosome'].astype('str') + "_" +
                self.dataset['Start position'].astype('str') + "_" +
                self.dataset['Reference'].astype('str') + "_" +
                self.dataset['Variant'].astype('str'))

        # create subdirectories
        for target_value in self.dataset['Target'].unique().tolist():
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
            partial(save_image, folder=folder, reference_genome_path=reference_genome_path),
            self.dataset[['Chromosome', 'Start position std', 'End position std',
                          'Variant type', 'Child BAM', 'Father BAM', 'Mother BAM',
                          'Key', 'Target']].values
        )

        pool.close()
        print("Saving images finished, time elapsed:", datetime.datetime.now() - start)
        sys.stdout.flush()

    def apply_model(self, output_path, models_cfg, reference_genome_path, n_jobs=-1, batch_size=1000):

        # apply models in parallel
        print("\nStart applying DeNovoCNN")

        if n_jobs == -1:
            # use all CPUs
            n_jobs = mp.cpu_count()

        self.dataset = self.dataset[self.dataset['Variant type'] == 'Substitution']
        dataset_batches = self.dataset[['Chromosome', 'Start position std', 'End position std',
                                        'Variant type', 'Child BAM', 'Father BAM', 'Mother BAM'
                                        ]].values.tolist()

        dataset_batches = [dataset_batches[x:x + batch_size] for x in range(0, len(dataset_batches), batch_size)]

        print(f"Using {n_jobs} CPUs")
        sys.stdout.flush()
        start = datetime.datetime.now()

        pool = mp.Pool(n_jobs)

        results = pool.map(
            partial(apply_model_batch, models_cfg=models_cfg, reference_genome_path=reference_genome_path),
            dataset_batches
        )

        pool.close()
        print("Applying DeNovoCNN finished, time elapsed:", datetime.datetime.now() - start)
        sys.stdout.flush()

        # flatten results
        results = [pred for sub_pred in results for pred in sub_pred]

        self.dataset['DeNovoCNN probability'] = [res[0] for res in results]

        self.dataset[['Chromosome', 'Start position',
                      'Reference', 'Variant', 'DeNovoCNN probability',
                      'Child coverage', 'Father coverage', 'Mother coverage',
                      'Child BAM', 'Father BAM', 'Mother BAM']].to_csv(output_path, sep='\t', index=False)


def get_variant_class(row):
    """
    Defining variant type (SNP, insertion, deletion)
    based on reference and alternative alleles

    Parameters:
    row: DataFrame row

    Returns:
    Substitution, Deletion, Insertion or Unknown
    """
    reference, variant = row['Reference'], row['Variant']

    if len(reference) == len(variant):
        return 'Substitution'
    elif len(reference) > len(variant):
        return 'Deletion'
    elif len(reference) < len(variant):
        return 'Insertion'

    return 'Unknown'

def generate_images_from_folder(folder, save_folder, reference_genome_path, convert_to_inner_format, n_jobs=-1):
    files = glob.glob(f"{folder}/*_*.csv")

    for dataset_path in files:
        print("\nLoading images for", dataset_path)
        dataset_type, variant_type = tuple(dataset_path.split('/')[-1].split('.')[0].split('_'))

        full_save_folder = os.path.join(save_folder, variant_type, dataset_type)

        dataset = Dataset(dataset=pd.read_csv(dataset_path, sep='\t'), convert_to_inner_format=convert_to_inner_format)

        dataset.save_images(folder=full_save_folder, reference_genome_path=reference_genome_path, n_jobs=n_jobs)


def remove_matching_nucleotides(row):
    """
    removes matching nucleotides in Reference and Variant from beginning and end,
    updates variant positions accordingly,
    saves the results in new columns
    :param row: row from a processed dataframe
    """
    row['Reference std'] = row['Reference']
    row['Variant std'] = row['Variant']
    row['Start position std'] = row['Start position']

    # remove from the end
    if len(row['Reference std']) == len(row['Variant std']):
        end_prefix = os.path.commonprefix([row['Reference std'][::-1], row['Variant std'][::-1]])

        row['Reference std'] = row['Reference std'][:(len(row['Reference std']) - len(end_prefix))]
        row['Variant std'] = row['Variant std'][:(len(row['Variant std']) - len(end_prefix))]

    # remove from the beginning
    prefix = os.path.commonprefix([row['Reference std'], row['Variant std']])

    row['Start position std'] += len(prefix)
    row['Reference std'] = row['Reference std'][len(prefix):]
    row['Variant std'] = row['Variant std'][len(prefix):]

    row['End position std'] = row['Start position std'] + max(len(row['Reference std']), 1)

    return row

# Changed to new encoding
def get_image(row, reference_genome_path):
    """
    creates TrioVariant class object for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param reference_genome_path: path to a reference genome
    """
    chromosome, start, end, _, cbam, fbam, mbam, key, target = tuple(row)

    return gen_trio_img(cbam, fbam, mbam, reference_genome_path, chromosome, start)


def save_image(row, folder, reference_genome_path):
    """
    saves RGB image for DeNovoCNN for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param folder: folder to place an image
    :param reference_genome_path: path to a reference genome
    """
    chromosome, start, end, var_type, child_bam, father_bam, mother_bam, key, target = tuple(row)

    trio_variant_image = gen_trio_img(child_bam, father_bam, mother_bam, reference_genome_path, chromosome, start)

    # swap dimensions
    swapped_image = np.zeros(trio_variant_image.shape)
    swapped_image[:, :, 0], swapped_image[:, :, 1], swapped_image[:, :, 2] = (
        trio_variant_image[:, :,2], trio_variant_image[:, :,1], trio_variant_image[:, :, 0])

    # save image
    output_path = os.path.join(folder, target, f"{key}.png")
    cv2.imwrite(output_path, swapped_image)

    return output_path


def load_models(models_cfg):
    """
    loads Substitution, Deletion and Insertion models from paths specified in models_cfg
    """
    models_dict = {
        'Substitution': tf.keras.models.load_model(models_cfg['snp_model'], compile=False),
        'Deletion': tf.keras.models.load_model(models_cfg['del_model'], compile=False),
        'Insertion': tf.keras.models.load_model(models_cfg['ins_model'], compile=False)
    }

    return models_dict


def predict(img, model):
    """
        Applies the model to RGB image
        and gets DNM prediction
    """
    expanded_image = np.expand_dims(img, axis=0)
    normalized_image = expanded_image.astype(float) / 255
    prediction = model.predict(normalized_image)
    return prediction

def apply_model(row, models_dict, reference_genome_path):
    """
    applies DeNovoCNN models from models_dict to a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param models_dict: contains information about DeNovoCNN models paths
    :param reference_genome_path: path to a reference genome
    """
    _, _, _, var_type, _, _, _ = tuple(row)

    try:
        trio_variant_image = get_image(tuple(row) + (None, None), reference_genome_path)
        # predict
        prediction = predict(trio_variant_image, models_dict[var_type])
        prediction_dnm = str(round(1. - prediction[0, 0], 3))

    except:
        print("Failed in:", flush=True)
        print("\t", row, flush=True)
        return -1, (-1, -1, -1)

    return prediction_dnm


def apply_model_batch(rows, models_cfg, reference_genome_path):
    """
    applies  DeNovoCNN models from models_cfg to a batch of rows from a processed dataframe
    """
    models_dict = load_models(models_cfg)
    return [apply_model(row, models_dict, reference_genome_path) for row in rows]


def apply_models_on_trio(variants_list, output_path, child_bam, father_bam, mother_bam,
                         snp_model, del_model, ins_model, ref_genome, convert_to_inner_format, n_jobs):
    """
    applies DeNovoCNN models in parallel to all variants specified in variants_list for a trio
    """

    trio_cfg = {
        'Child': {'bam_path': child_bam},
        'Father': {'bam_path': father_bam},
        'Mother': {'bam_path': mother_bam}
    }

    models_cfg = {
        'snp_model': snp_model,
        'del_model': del_model,
        'ins_model': ins_model
    }

    # read variants list
    variant_cols = ['Chromosome', 'Start position', 'Reference', 'Variant', 'extra']

    dataset = pd.read_csv(variants_list, sep='\t', names=variant_cols)

    # fill in sample information
    for sample in trio_cfg:
        dataset[sample] = trio_cfg[sample].get('id', '')
        dataset[f"{sample} BAM"] = trio_cfg[sample]['bam_path']

    print(f"Start apply in parallel, n_jobs={n_jobs}")

    # apply models
    dataset = Dataset(dataset=dataset, convert_to_inner_format=convert_to_inner_format)

    dataset.apply_model(
        output_path=output_path,
        models_cfg=models_cfg,
        reference_genome_path=ref_genome,
        n_jobs=n_jobs,
        batch_size=1000)

