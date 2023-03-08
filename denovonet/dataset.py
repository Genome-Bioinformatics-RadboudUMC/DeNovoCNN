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
from denovonet.variants import SingleVariant, TrioVariant
from denovonet.encoders import baseEncoder


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

        # flattern results
        results = [pred for sub_pred in results for pred in sub_pred]

        self.dataset['DeNovoCNN probability'] = [res[0] for res in results]
        self.dataset['Child coverage'] = [res[1][0] for res in results]
        self.dataset['Father coverage'] = [res[1][1] for res in results]
        self.dataset['Mother coverage'] = [res[1][2] for res in results]

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
        return SingleVariant(str(chromosome), int(start), int(end), bam, reference_genome)
    except (ValueError, KeyError):
        if chromosome[:3] != 'chr':
            chromosome = 'chr' + chromosome
            return SingleVariant(str(chromosome), int(start), int(end), bam, reference_genome)
        elif chromosome[:3] == 'chr':
            chromosome = chromosome[:3]
            return SingleVariant(str(chromosome), int(start), int(end), bam, reference_genome)
        else:
            raise

# New Image encoding:
def parse_insertion(line):
    if '+' not in line:
        return []

    line = line.split('+')[1]
    inserted_bases = ['IN_' + i for i in line if not i.isdigit()]

    return inserted_bases


def merge_lists(lists):
    merged_lists = np.empty([len(lists), len(max(lists, key=lambda lst: len(lst)))], dtype='U4')

    for i, lst in enumerate(lists):
        merged_lists[i][0:len(lst)] = lst

    return merged_lists


def process_pileup(cur_sequences, middle):
    cur_sequences['is_snp'] = ~(
            cur_sequences['annot'].str.contains(',') |
            cur_sequences['annot'].str.contains('\.')
    )

    cur_sequences['is_del'] = cur_sequences['annot'].str.contains('\*')

    cur_sequences['is_ins'] = cur_sequences['annot'].str.contains('\+')

    # get insertions
    cur_sequences['insertion'] = (
        merge_lists(cur_sequences['annot'].str.upper().apply(parse_insertion).values).tolist()
    )

    ins_length = len(cur_sequences['insertion'].iloc[0])

    # process base
    cur_sequences['processed_base'] = cur_sequences['raw'].str.upper()
    cur_sequences.loc[cur_sequences['annot'].str.contains('\*'), 'processed_base'] = 'DEL'

    # process qualities

    if cur_sequences.loc[~cur_sequences['is_del']].shape[0] > 0:
        cur_sequences.loc[cur_sequences['is_del'], 'baseq'] = cur_sequences.loc[
            ~cur_sequences['is_del'], 'baseq'].mean()
    else:
        cur_sequences.loc[cur_sequences['is_del'], 'baseq'] = 60.

    is_middle = (cur_sequences['position'] == middle)

    cur_sequences['processed_qual'] = cur_sequences['mapq'] * cur_sequences['baseq'] // 10
    cur_sequences.loc[~cur_sequences['is_snp'] & ~cur_sequences['is_del'] & ~is_middle, 'processed_qual'] = (
            cur_sequences.loc[~cur_sequences['is_snp'] & ~cur_sequences['is_del'] & ~is_middle, 'processed_qual'] // 3
    )

    cur_sequences['insertions_qual'] = cur_sequences['mapq'] * cur_sequences['baseq'] // 10

    cur_sequences.loc[~cur_sequences['is_ins'] & ~is_middle, 'insertions_qual'] = (
            cur_sequences.loc[~cur_sequences['is_ins'] & ~is_middle, 'insertions_qual'] // 3
    )

    # get bases

    bases = np.hstack([cur_sequences[['processed_base']].values,
                       np.array(cur_sequences.insertion.values.tolist())])

    qualities = np.hstack([cur_sequences[['processed_qual']].values,
                           cur_sequences[['insertions_qual'] * ins_length].values])

    insertions_mask = np.hstack([np.zeros((1, 1)), np.ones((1, ins_length))])

    return cur_sequences.index, bases, qualities, insertions_mask


def get_encodings(encoding):
    encoding_match = {
        'DEL': baseEncoder.DEL,
        'A': baseEncoder.A,
        'C': baseEncoder.C,
        'T': baseEncoder.T,
        'G': baseEncoder.G,
        'IN_A': baseEncoder.IN_A,
        'IN_C': baseEncoder.IN_C,
        'IN_T': baseEncoder.IN_T,
        'IN_G': baseEncoder.IN_G,
        'IN_N': baseEncoder.IN_A,
    }

    return encoding_match.get(encoding, baseEncoder.EMPTY)


get_encodings_vectorized = np.vectorize(get_encodings)


def update_query_names(query_names_df, new_query_names):
    """
        Keep names of the reads in a right order.
    """
    query_names_df = pd.concat([query_names_df,
                                pd.DataFrame(new_query_names, columns=['query_name'])])

    query_names_df = query_names_df.drop_duplicates(subset='query_name', keep='first').reset_index(drop=True)
    query_names_df['idx'] = query_names_df.index

    return query_names_df


def get_single_variant_encodings(bam, middle, chromosome, ref_path):
    reference_genome = pysam.FastaFile(ref_path)
    middle -= 1
    query_names_df = pd.DataFrame([])

    pileup_encoded = np.empty((160, 41 * 100), dtype='U4')
    quality_encoded = np.zeros((160, 41 * 100), dtype='int')
    insertions_mask_encoded = np.zeros((1, 41 * 100), dtype='int')
    col_pointer = 0
    positions = []

    for pileup_column in bam.pileup(contig=chromosome, start=middle - 20, stop=middle + 20 + 1,
                                    truncate=True, min_base_quality=0, fastafile=reference_genome):
        cur_query_names = pileup_column.get_query_names()

        query_names_df = update_query_names(query_names_df=query_names_df,
                                            new_query_names=cur_query_names)

        cur_sequences = pd.DataFrame([pileup_column.get_mapping_qualities(),
                                      pileup_column.get_query_qualities(),
                                      pileup_column.get_query_sequences(),
                                      pileup_column.get_query_sequences(mark_matches=True, add_indels=True),
                                      ], columns=cur_query_names, index=['mapq', 'baseq', 'raw', 'annot']
                                     ).transpose()
        #         return cur_sequences

        cur_sequences['position'] = pileup_column.reference_pos

        indexes, bases, qualities, insertions_mask = process_pileup(cur_sequences, middle)
        indexes = query_names_df.set_index('query_name').loc[indexes.values, 'idx']

        _, col_added = bases.shape

        pileup_encoded[indexes, col_pointer:col_pointer + col_added] = bases
        quality_encoded[indexes, col_pointer:col_pointer + col_added] = qualities
        insertions_mask_encoded[0:1, col_pointer:col_pointer + col_added] = insertions_mask

        positions += [cur_sequences['position'].iloc[0]] * col_added

        col_pointer += col_added

    if len(positions) == 0:
        positions = range(middle - 20, middle + 20 + 1)
    middle_col = positions.index(middle)

    pileup_encoded = pileup_encoded[:, middle_col - 20:middle_col + 20 + 1]
    quality_encoded = quality_encoded[:, middle_col - 20:middle_col + 20 + 1]
    insertions_mask_encoded = insertions_mask_encoded[:, middle_col - 20:middle_col + 20 + 1]
    positions = positions[middle_col - 20:middle_col + 20 + 1]

    pileup_encoded = get_encodings_vectorized(pileup_encoded)
    quality_encoded = quality_encoded.astype(int)
    quality_encoded[quality_encoded > 255] = 255

    return pileup_encoded, quality_encoded, positions


# def align_insertions(child, father, mother):
def update_positions(positions):
    new_positions = []

    counter = 0
    for idx, pos in enumerate(positions):

        if idx == 0:
            new_positions.append(str(pos))
            continue

        if pos == positions[idx - 1]:
            counter += 1
            new_positions.append(str(pos) + "_" + str(counter))
        else:
            counter = 0
            new_positions.append(str(pos))

    return new_positions


class SimpVar:
    def __init__(self, pileup, quality, positions, start_coverage):
        self.pileup_encoded = pileup
        self.quality_encoded = quality
        self.positions = update_positions(positions)
        self.start_coverage = start_coverage


def align_data(child_var, father_var, mother_var, start):
    child_df = pd.DataFrame(child_var.pileup_encoded, columns=child_var.positions)
    child_df['tag'] = 'child_pileup'

    mother_df = pd.DataFrame(mother_var.pileup_encoded, columns=mother_var.positions)
    mother_df['tag'] = 'mother_pileup'

    father_df = pd.DataFrame(father_var.pileup_encoded, columns=father_var.positions)
    father_df['tag'] = 'father_pileup'

    child_df_q = pd.DataFrame(child_var.quality_encoded, columns=child_var.positions)
    child_df_q['tag'] = 'child_qual'

    mother_df_q = pd.DataFrame(mother_var.quality_encoded, columns=mother_var.positions)
    mother_df_q['tag'] = 'mother_qual'

    father_df_q = pd.DataFrame(father_var.quality_encoded, columns=father_var.positions)
    father_df_q['tag'] = 'father_qual'

    merged_df = pd.concat([child_df, mother_df, father_df, child_df_q, mother_df_q, father_df_q]).fillna(0)

    positions_order = sorted([col for col in merged_df.columns if col != 'tag'])
    middle_col = positions_order.index(str(start))
    positions_order = positions_order[middle_col - 20:middle_col + 20 + 1]

    child_var.pileup_encoded = merged_df[merged_df['tag'] == 'child_pileup'][positions_order].values
    father_var.pileup_encoded = merged_df[merged_df['tag'] == 'father_pileup'][positions_order].values
    mother_var.pileup_encoded = merged_df[merged_df['tag'] == 'mother_pileup'][positions_order].values

    child_var.quality_encoded = merged_df[merged_df['tag'] == 'child_qual'][positions_order].values
    father_var.quality_encoded = merged_df[merged_df['tag'] == 'father_qual'][positions_order].values
    mother_var.quality_encoded = merged_df[merged_df['tag'] == 'mother_qual'][positions_order].values

    child_var.positions = positions_order
    father_var.positions = positions_order
    mother_var.positions = positions_order

    return child_var, father_var, mother_var

def count_coverage(bam_data, chromosome, pos):
    start_coverage_arrays = bam_data.count_coverage(chromosome, pos - 1, pos)
    return sum([coverage[0] for coverage in start_coverage_arrays])

# Regular DeNovoCNN

def get_image(row, reference_genome_path):
    """
    creates TrioVariant class object for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param reference_genome_path: path to a reference genome
    """
    chromosome, start, end, _, cbam, fbam, mbam, key, target = tuple(row)

    child_bam = pysam.AlignmentFile(cbam)
    father_bam = pysam.AlignmentFile(fbam)
    mother_bam = pysam.AlignmentFile(mbam)
    # create image with new encoding
    child_var = SimpVar(*get_single_variant_encodings(child_bam, start, chromosome, reference_genome_path),
                        count_coverage(child_bam, chromosome, start))
    father_var = SimpVar(*get_single_variant_encodings(father_bam, start, chromosome, reference_genome_path),
                         count_coverage(father_bam, chromosome, start))
    mother_var = SimpVar(*get_single_variant_encodings(mother_bam, start, chromosome, reference_genome_path),
                         count_coverage(mother_bam, chromosome, start))

    child_var, father_var, mother_var = align_data(child_var, father_var, mother_var, start)
    trio = TrioVariant(child_var, father_var, mother_var)

    return trio


def save_image(row, folder, reference_genome_path):
    """
    saves RGB image for DeNovoCNN for a corresponding row from a processed dataframe
    :param row: row from a processed dataframe
    :param folder: folder to place an image
    :param reference_genome_path: path to a reference genome
    """
    chromosome, start, end, var_type, child_bam, father_bam, mother_bam, key, target = tuple(row)

    trio_variant = get_image(row, reference_genome_path)

    # swap dimensions
    swapped_image = np.zeros(trio_variant.image.shape)
    swapped_image[:, :, 0], swapped_image[:, :, 1], swapped_image[:, :, 2] = (
        trio_variant.image[:, :,2], trio_variant.image[:, :,1], trio_variant.image[:, :, 0])

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
        prediction_dnm = str(round(1. - prediction[0, 0], 3))

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

