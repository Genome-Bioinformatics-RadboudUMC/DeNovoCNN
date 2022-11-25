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

import argparse
from denovonet.dataset import apply_models_on_trio, apply_models_on_trio_onnx

# parse arguments
parser = argparse.ArgumentParser(description='Use DeNovoCNN.')

parser.add_argument('--mode', default='predict', type=str,
                    help='Mode that is used to run DeNovoCNN. Possible modes:\ntrain\npredict')

# predict arguments
parser.add_argument('--variants_list', dest='variants_list',
                    type=str, help='Path to a file with list of variants to check with DeNovoCNN')

parser.add_argument('--snp_model', dest='snp_model',
                    type=str, help='Path to substitutions model')

parser.add_argument('--del_model', dest='del_model',
                    type=str, help='Path to deletions model')

parser.add_argument('--ins_model', dest='ins_model',
                    type=str, help='Path to insertions model')

parser.add_argument('--child_bam', dest='child_bam',
                    type=str, help='Path to child BAM file')

parser.add_argument('--father_bam', dest='father_bam',
                    type=str, help='Path to father BAM file')

parser.add_argument('--mother_bam', dest='mother_bam',
                    type=str, help='Path to mother BAM file')

parser.add_argument('--ref_genome', dest='ref_genome',
                    type=str, help='Path to reference genome file.')

parser.add_argument('--output_denovocnn_format', dest='output_denovocnn_format', type=str,
                    help='Should be true or false, default: false. ' +
                         'If set to true, the output file will contain normalized variants and end coordinate')

parser.add_argument('--output_path', dest='output_path', default='output.txt', type=str,
                    help='Path to output file with DeNovoCNN predictions.')

args = parser.parse_args()

# run DeNovoCNN training or predicting
if __name__ == "__main__":
    
    # training of the DeNovoCNN
    if args.mode == 'train':
        # model = models.train(EPOCHS, IMAGES_FOLDER, DATASET_NAME, output_model_path)
        pass
    
    # prediction of the DeNovoCNN
    elif args.mode == 'predict':
        apply_models_on_trio(
            variants_list=args.variants_list,
            output_path=args.output_path,
            child_bam=args.child_bam,
            father_bam=args.father_bam,
            mother_bam=args.mother_bam,
            snp_model=args.snp_model,
            del_model=args.del_model,
            ins_model=args.ins_model,
            ref_genome=args.ref_genome,
            output_denovocnn_format=args.output_denovocnn_format,
            convert_to_inner_format=True,
            n_jobs=-1
        )
    # prediction of the DeNovoCNN
    elif args.mode == 'predict_onnx':
        apply_models_on_trio_onnx(
            variants_list=args.variants_list,
            output_path=args.output_path,
            child_bam=args.child_bam,
            father_bam=args.father_bam,
            mother_bam=args.mother_bam,
            snp_model=args.snp_model,
            del_model=args.del_model,
            ins_model=args.ins_model,
            ref_genome=args.ref_genome,
            output_denovocnn_format=args.output_denovocnn_format,
            convert_to_inner_format=True,
            n_jobs=-1
        )

    else:
        print('Error. Unknown mode: {} . Please choose one of the following:\ntrain\npredict'.format(args.mode))