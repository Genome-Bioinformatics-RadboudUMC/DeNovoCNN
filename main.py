'''
main.py

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
import argparse

from denovonet.dataset import Dataset
from denovonet import models
from denovonet.infer import infer_dnms_from_intersected

from keras.models import load_model

import pysam

# parse arguments
parser = argparse.ArgumentParser(description='Run denovoCNN on a trio.')
parser.add_argument('--mode', default='train', type=str, help='Mode that is used to run DeNovoNet. Possible modes:\ntrain\npredict')

# train mode arguments
parser.add_argument('--build-dataset', dest='build_dataset', action='store_true', 
                    help='Build and save new images dataset that will be used for training.')
parser.add_argument('--use-dataset', dest='build_dataset', action='store_false', 
                    help='Use existing images dataset for training.')
parser.add_argument('--continue-training', dest='continue_training', action='store_true', 
                    help='Continue training model - used for transfer learning, --model-path should be specified if used.')

parser.add_argument('--train-dataset', dest='train_dataset', default=None, type=str, 
                    help='Path to TSV file that is used to build training data')
parser.add_argument('--val-dataset', dest='val_dataset', default=None, type=str, 
                    help='Path to TSV file that is used to build val data')
parser.add_argument('--images', default=None, type=str, 
                    help='Path to folder that contains images for training ot will be used to save images')
parser.add_argument('--dataset-name', dest='dataset_name', default=None, type=str, 
                    help='Name of the dataset.')
parser.add_argument('--epochs', default=45, type=int, help='Name of epochs for training.')

parser.add_argument('--model-path', dest='model_path', default=None, type=str, 
                    help='Path to model for training mode with --continue-training.')
parser.add_argument('--output-model-path', dest='output_model_path', default=None, type=str, 
                    help='Output path for model in training mode.')

# train or predict mode arguments
parser.add_argument('--genome', default=None, type=str, help='Path to reference genome file.')

# predict mode arguments
parser.add_argument('--child-bam', dest='child_bam', default=None, type=str, help='Path to child BAM.')
parser.add_argument('--father-bam', dest='father_bam', default=None, type=str, help='Path to father BAM.')
parser.add_argument('--mother-bam', dest='mother_bam', default=None, type=str, help='Path to mother BAM.')

parser.add_argument('--variants-list', dest='variants_list', default='intersected.txt', type=str, 
                    help='Path to variants list file.')

parser.add_argument('--snp-model', dest='snp_model', default=None, type=str, help='Path to SNP model.')
parser.add_argument('--in-model', dest='in_model', default=None, type=str, help='Path to insertion model.')
parser.add_argument('--del-model', dest='del_model', default=None, type=str, help='Path to deletion model.')

parser.add_argument('--create-filtered-file', dest='create_filtered_file', default="false", type=str, 
                    help="True/False to create a filtered (predictions >= 0.5) file")

parser.add_argument('--output', default='output.txt', type=str, 
                    help='Path to output file with DeNovoCNN predictions.')
args = parser.parse_args()

# set arguments
EPOCHS = args.epochs
IMAGES_FOLDER = args.images
DATASET_NAME = args.dataset_name

# run DeNovoCNN training or predicting
if __name__ == "__main__":
    
    # training of the DeNovoCNN
    if args.mode == 'train':
        
        output_model_path = args.output_model_path
        REREFERENCE_GENOME = pysam.FastaFile(args.genome)
        
        
        if args.build_dataset:
            # create and save images dataset for training and validation
            
            train_variants_path = args.train_dataset
            val_variants_path = args.val_dataset

            print('Building training dataset based on file {}'.format(train_variants_path))
            train_dataset = Dataset(train_variants_path, 'train', REREFERENCE_GENOME)
            
            print('Building validation dataset based on file {}'.format(val_variants_path))
            val_dataset = Dataset(val_variants_path, 'val', REREFERENCE_GENOME)
            
            train_dataset.save_images(IMAGES_FOLDER, DATASET_NAME)
            val_dataset.save_images(IMAGES_FOLDER, DATASET_NAME)
        
        # training the model
        if args.continue_training:
            # continue training using model in --model_path
            print('Continuing training model {} .'.format(args.model_path))
            model = models.train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path,
                                 continue_training=True,input_model_path=args.model_path)
        else:
            # training the model from scratch
            print('Training new model.')
            model = models.train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path)
    
    # prediction of the DeNovoCNN
    elif args.mode == 'predict':
        REREFERENCE_GENOME = pysam.FastaFile(args.genome)

        child_bam = args.child_bam
        father_bam = args.father_bam
        mother_bam = args.mother_bam

        snp_model = args.snp_model
        in_model = args.in_model
        del_model = args.del_model

        path_to_tsv = args.variants_list
        output_path = args.output

        create_filtered_file = args.create_filtered_file.lower() in ('true', 't')
        
        # run DeNovoCNN on variants list in path_to_tsv and saving the results in output_path
        dnms_table = infer_dnms_from_intersected(path_to_tsv, child_bam, father_bam, mother_bam, 
                                                 REREFERENCE_GENOME, snp_model, in_model, del_model)

        dnms_table.to_csv(output_path, sep='\t',index=False)

        print('Full analysis saved as {}.'.format(output_path))
        
        # save DeNovoCNN predicted DNM variants with probability >=0.5 in separate file
        if create_filtered_file:
            dnms_only_table = dnms_table.loc[dnms_table['DNV probability'] >= 0.5]
            dnms_only_table.to_csv(output_path + '.filtered.txt', sep='\t',index=False)
            print('Predicted DNVs (probability >= 0.5) saved as {}.'.format(output_path + '.filtered.txt'))

    else:
        print('Error. Unknown mode: {} . Please choose one of the following:\ntrain\npredict'.format(args.mode))