'''
infer.py

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

from denovonet.settings import MINIMAL_COVERAGE
from denovonet.settings import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, MODEL_ARCHITECTURE, CLASS_MODE, NUMBER_CLASSES

from denovonet.encoders import VariantClassValue, VariantInheritance
from denovonet.variants import SingleVariant, TrioVariant
from denovonet.models import get_model
from denovonet.logger import call_logger, entering, exiting

from keras.models import load_model
from keras import backend as K

import time
import numpy as np
import pandas as pd
import os
import sys
import subprocess
from difflib import SequenceMatcher
import math

def get_variant_class(reference, alternate):
    """
    Defining variant type (SNP, insertion, deletion)
    based on reference and alternative alleles
    
    Parameters:
    reference (str): reference allele
    alternate (str): alternative allele
    
    Returns: 
    VariantClassValue: snp, deletion, insertion or unknown
    """
    if len(reference) == 1 and len(alternate) == 1:
        return VariantClassValue.snp
    elif len(reference) > len(alternate):
        return VariantClassValue.deletion
    elif len(reference) < len(alternate):
        return VariantClassValue.insertion
    else:
        if ',' in alternate:
            return VariantClassValue.deletion
        elif ',' in reference:
            return VariantClassValue.insertion
        else:
            return VariantClassValue.unknown

        
def get_end_coordinate(reference, start):
    """
    Calculate variant end position based on 
    reference allele and variant start position
    
    Parameters:
    reference (str): reference allele
    start (str or int): variant start position
    
    Returns: 
    str: variant end position
    """
    return str( int(start) + len(reference) - 1 )


def remove_matching_string(start, ref, var):
    """
    Remove common prefix in reference and alterative 
    alleles updating start position of the variant
    
    Parameters:
    start (int): variant start position
    ref (str): reference allele
    var (str): alternative allele
    
    Returns: 
    int: updated variant start position
    str: reference allele without common prefix
    str: alternative allele without common prefix
    """
    insertion = bool(len(ref) > len(var))
    
    match = SequenceMatcher(None, ref, var, 
                            autojunk=False).find_longest_match(0, len(ref), 0, len(var))
    

    new_ref = ref.replace(ref[match.a: match.a + match.size], '', 1)
    new_var = var.replace(var[match.b:  match.b + match.size], '', 1)

    if insertion:
        start += match.size
    
    if match.size == 0:
        return start, new_ref, new_var
    else:
        return remove_matching_string(start, new_ref, new_var)


def check_chromosome(chromosome):
    """
    Check if chromosome in 
    ['1', ..., '23', 'X', 'Y', MT', 'M'] or
    ['chr1', ..., 'chr23', 'chrX', 'chrY', chrMT', 'chrM']
    
    Parameters:
    chromosome (str): chromosome 
    
    Returns: 
    str: None if chromosome not in accepted lists, otherwise chromosome
    """
    
    accepted_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT', 'M']
    accepted_chromosomes += ['chr'+ i for i in accepted_chromosomes]

    chromosome = str(chromosome)
        
    if chromosome not in accepted_chromosomes:
        print ('Skipping chromosome (bad naming):', chromosome)
        return None
        
    return chromosome

def split_comma_separated_variants(intersected_dataframe):
    """
    Preprocessing step for input dataframe of variants 
    
    Parameters:
    intersected_dataframe (pd.DataFrame): dataframe containing variants 
    
    Returns: 
    np.array: preprocessed list of variants
    """
    row_list = []

    for row in np.array(intersected_dataframe):
        chromosome, start, ref, var, extra = row
         
        if check_chromosome(chromosome) is None:
            continue
           
        if (ref is None):
            ref = ''
            
        if (var is None):
            var = ''
        
        for splitted_variant in var.split(','):

            new_start, new_ref, new_splitted_variant = remove_matching_string(start, ref, splitted_variant)          
            new_row = chromosome, new_start, new_ref, new_splitted_variant, extra
            row_list.append(new_row)

    return np.array(row_list)

def infer_dnms_from_intersected(intersected_variants_tsv, child_bam, father_bam, mother_bam, 
                                REREFERENCE_GENOME, snp_model, in_model, del_model):
    """
    DeNovoCNN predicted probabilities for DNM class for the variants in intersected_variants_tsv
    
    Parameters:
    intersected_variants_tsv (str): path to the file with variants list
    child_bam (str): path to the child BAM file
    father_bam (str): path to the father BAM file
    mother_bam (str): path to the mother BAM file
    REFERENCE_GENOME (pysam.FastaFile): reference genome FastaFile object
    snp_model (str): path to the SNPs model file
    in_model (str): path to the insertions model file
    del_model (str): path to the deletions model file
    
    Returns: 
    pd.DataFrame: DataFrame with predicted DNM probabilities for the input list of variants
    """

    print('SNP model', snp_model)
    print('Insertion model', in_model)
    print('Deletion model', del_model)
    
    # load models 
    # with this approach to avoid pytorch versions compatibility conflict 
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    
    model_snps = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    model_insertions = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    model_deletions = get_model(MODEL_ARCHITECTURE, input_shape, NUMBER_CLASSES)
    
    model_snps.load_weights(snp_model)
    model_insertions.load_weights(in_model)
    model_deletions.load_weights(del_model)
    
    # load and preprocess list of variants
    intersected_dataframe = pd.read_csv(intersected_variants_tsv, sep='\t', 
                                        names=['Chromosome','Position','Reference','Variant','extra'])
    intersected_dataframe = intersected_dataframe.replace(np.nan, '', regex=True)
    print(intersected_dataframe.head())
    
    splitted_intersected_array = split_comma_separated_variants(intersected_dataframe)
    
    # iterate over list of variants, create an RGB image, predict DNM probability
    start_time = time.time()
    dnms_table = []
    
    for counter, variant in enumerate(splitted_intersected_array):
        if counter % 100 == 0:
            elapsed = round((time.time() - start_time), 0)
            print('Variants evaluated: {} . Total time elapsed: {}s'.format(counter, str(elapsed)))
            sys.stdout.flush()
        
        # retrieve and preprocess variant information
        chromosome, start, reference, alternate, _ = tuple(variant)
        start = str(start)
        variant_class = get_variant_class(reference, alternate)
        end = get_end_coordinate(reference, start)

        if variant_class == VariantClassValue.deletion or variant_class == VariantClassValue.insertion:
            if variant_class == VariantClassValue.insertion:
                end = str(int(end) + 1)

        # encoding variant location as numpy arrays for each sample
        child_variant = SingleVariant(str(chromosome), int(start), int(end)+1, child_bam, REREFERENCE_GENOME)
        father_variant = SingleVariant(str(chromosome), int(start), int(end)+1, father_bam, REREFERENCE_GENOME)
        mother_variant = SingleVariant(str(chromosome), int(start), int(end)+1, mother_bam, REREFERENCE_GENOME)
        
        if ((child_variant.start_coverage < MINIMAL_COVERAGE) or (father_variant.start_coverage < MINIMAL_COVERAGE) 
            or (mother_variant.start_coverage < MINIMAL_COVERAGE)):
            # skip low coverage regions
            dnms_table_row = [chromosome, start, end, reference, alternate, -1]
            dnms_table.append(dnms_table_row)
            continue
        elif variant_class == VariantClassValue.unknown:
            # skip unknown variant type
            dnms_table_row = [chromosome, start, end, reference, alternate, -2]
            dnms_table.append(dnms_table_row)
            continue
        else: # apply DeNovoCNN
            
            mean_start_coverage = (child_variant.start_coverage + father_variant.start_coverage + mother_variant.start_coverage) / 3
            mean_start_coverage = int(round(mean_start_coverage))
            
            # encode variant as RGB image
            trio_variant = TrioVariant(child_variant, father_variant, mother_variant)
            
            # make and preprocess prediction using DeNovoCNN
            if variant_class == VariantClassValue.snp:
                prediction = trio_variant.predict(model_snps)
            elif variant_class == VariantClassValue.deletion:
                prediction = trio_variant.predict(model_deletions)
            elif variant_class == VariantClassValue.insertion:
                prediction = trio_variant.predict(model_insertions)
            else:
                prediction_dnm = np.array([-2,-2])
            
            argmax = np.argmax(prediction, axis=1)
            
            if CLASS_MODE == 'binary':
                prediction_dnm = str(round(1.-prediction[0,0],3))
            else:
                prediction_dnm = str(round(prediction[0,0],3))
            
            # write prediction and coverage to results table
            dnms_table.append([chromosome, start, end, reference, alternate, 
                               float(prediction_dnm), mean_start_coverage])
    
    K.clear_session()
    
    # make resulting DataFrame with DNM probabilities
    dnms_table = pd.DataFrame(dnms_table, columns=['Chromosome','Start position','End position','Reference','Variant','DNV probability','Mean Start Coverage'])
    return dnms_table
