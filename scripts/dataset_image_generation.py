import pysam
import numpy as np
import pandas as pd
import glob
import os
from denovonet.variants import SingleVariant, TrioVariant, SingleVariantLRS, TrioVariantLRS
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from functools import partial
import glob
import os
from collections import defaultdict
import datetime

# function to access a specific .bam file based on a dna-id
def get_revio_bam_path(dna_id):
    files_lst = (
        glob.glob(f'/ifs/data/research/revio/work/{dna_id}/GRCh38*/{dna_id}*.bam') + 
        glob.glob(f'/ifs/data/research/revio/work/{dna_id}/GRCh38*/BAMs_*/{dna_id}*.bam')
    )

    if len(files_lst) == 0: raise FileNotFoundError(f"No bam files found for {dna_id}")
    if len(files_lst) > 1: files_lst = sorted(files_lst, key=os.path.getmtime)

    return files_lst[-1]

def infer_variant_type(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1: return "snp"
    if len(ref) == 1 and len(alt) > 1: return "ins"
    if len(ref) > 1 and len(alt) == 1:  return "del"

    return "complex"

trios = pd.read_csv("/ifs/data/research/revio/work/familyInfo.txt", sep= '\t')

dataset = pd.read_csv('/ifs/data/research/projects/gelana/denovocnn_lrs/data/revio_dnm/training_dataset.tsv', sep='\t')
dataset['variant_type'] = dataset[['ref', 'alt']].apply(lambda x: infer_variant_type(x.ref, x.alt), axis=1)
dataset = dataset[dataset['DNM_status']!='UNKNOWN']

# gets the id-s of all children
children = dataset['child'].values.tolist()
ids = np.unique(children)

# split the data into train, validation and test
train_ids, remain = train_test_split(ids, test_size=0.3, random_state=42)# 70% train
val_ids, test_ids = train_test_split(remain, test_size=0.5, random_state=42)# 15% validation and 15% test

snp_train = dataset[(dataset['child'].isin(train_ids)) & (dataset['variant_type'] == 'snp')]
snp_val = dataset[(dataset['child'].isin(val_ids)) & (dataset['variant_type'] == 'snp')]
snp_test = dataset[(dataset['child'].isin(test_ids)) & (dataset['variant_type'] == 'snp')]

ins_train = dataset[(dataset['child'].isin(train_ids)) & (dataset['variant_type'] == 'ins')]
ins_val = dataset[(dataset['child'].isin(val_ids)) & (dataset['variant_type'] == 'ins')]
ins_test = dataset[(dataset['child'].isin(test_ids)) & (dataset['variant_type'] == 'ins')]

del_train = dataset[(dataset['child'].isin(train_ids)) & (dataset['variant_type'] == 'del')]
del_val = dataset[(dataset['child'].isin(val_ids)) & (dataset['variant_type'] == 'del')]
del_test = dataset[(dataset['child'].isin(test_ids)) & (dataset['variant_type'] == 'del')]


# create a summary table of the dataset splits
columns = {'train': [len(snp_train), len(ins_train), len(del_train), (len(snp_train) + len(ins_train) + len(del_train))],
           'validation': [len(snp_val), len(ins_val), len(del_val), (len(snp_val) + len(ins_val) + len(del_val))],
           'test': [len(snp_test), len(ins_test), len(del_test), (len(snp_test) + len(ins_test) + len(del_test))]}
indexes = ['substitution', 'insertion', 'deletion', 'total']

split_ids = [train_ids, val_ids, test_ids]
split_folders = ["train", "val", "test"]

dataset_table = pd.DataFrame(data=columns, index=indexes)

print("Number of variants before image generation:")
print (dataset_table.to_string())
print()

# define image generation function (cauion - uses global variables)
def generate_images_per_trio(child_id, dataset_type, images_save_path): 
    try:
        child_path = get_revio_bam_path(trios.loc[trios['child'] == child_id, 'child'].iloc[0])
        mother_path = get_revio_bam_path(trios.loc[trios['child'] == child_id, 'father'].iloc[0])
        father_path = get_revio_bam_path(trios.loc[trios['child'] == child_id, 'mother'].iloc[0])
    
        # separate the dataset based on the type of the variant
        variant_types_folders = ['substitution', 'insertion', 'deletion']
        snp = dataset[(dataset['variant_type'] == 'snp')]
        ins = dataset[(dataset['variant_type'] == 'ins')]
        dels = dataset[(dataset['variant_type'] == 'del')]
        variant_types = [snp, ins, dels]
        
        for sub_dataset, variant_type_folder in zip(variant_types, variant_types_folders) :
            # print ("Processing", variant_type_folder, flush=True)
            # separate DNMs and unknown
            possible_DNMs = sub_dataset[(sub_dataset['DNM_status'] == "POSSIBLY_PHASED_DNM") & (sub_dataset['child'] == child_id)]
            ivs = sub_dataset[(sub_dataset['DNM_status'] == "POSSIBLY_NOT_DNM") & (sub_dataset['child'] == child_id)]
            # print(len(possible_DNMs))
            
            # print(len(ivs))
            
            # generate DNM images
            for row in range(len(possible_DNMs)):
                sample = possible_DNMs.iloc[row]
                
                child = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, child_path, reference_genome)
                father = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, father_path, reference_genome)
                mother = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, mother_path, reference_genome)
                trio = TrioVariantLRS(child, father, mother)
                img_save_path = f"{images_save_path}/{variant_type_folder}/{dataset_type}/DNMs/{child_id}_{sample['chrom']}_pos{sample['pos']}.png"
                #print (img_save_path, flush=True)
                TrioVariantLRS.save_image(img_save_path, np.flip(trio.image,2))
            
            # generate IVs images
            for row in range(len(ivs)):
                sample = ivs.iloc[row]
                
                child = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, child_path, reference_genome)
                father = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, father_path, reference_genome)
                mother = SingleVariantLRS(sample['chrom'], int(sample['pos']), int(sample['pos']) + 1, mother_path, reference_genome)

                trio = TrioVariantLRS(child, father, mother)
                img_save_path = f"{images_save_path}/{variant_type_folder}/{dataset_type}/IVs/{child_id}_{sample['chrom']}_pos{sample['pos']}.png"
                #print (img_save_path, flush=True)
                TrioVariantLRS.save_image(img_save_path, np.flip(trio.image,2))
    except Exception as e:
        print(f"Problem with {child_id} from {dataset_type} with error: {str(e)}")
    
    return None

# Multiprocess image eneration
reference_genome = pysam.FastaFile('/ifs/data/research/projects/ina/test_trio/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.masked.fa')
images_save_path = "/ifs/data/research/projects/ina/DeNovoCNN_project/data_images"

start_t = datetime.datetime.now()

# Generate RGB images
for ids in range(len(split_ids)):
    dataset_type = split_folders[ids]
    generate_images_per_trio_multiprocess = partial(generate_images_per_trio, dataset_type=dataset_type, images_save_path=images_save_path)
    pool = mp.Pool(mp.cpu_count() - 1)
    
    _ = pool.map(generate_images_per_trio_multiprocess, split_ids[ids])
    
    pool.close()

print ("Elapsed time for image generation:", start_t - datetime.datetime.now())
print()

#Calculate output images
print ("Generated images statistics...")
pattern = f"{images_save_path}/*/*/*/*.png"
counts = defaultdict(int)

for filepath in glob.glob(pattern):
    directory = os.path.dirname(filepath)
    counts[directory] += 1
    
for directory, count in sorted(counts.items()):
    print(f"{directory}: {count} png files")
    