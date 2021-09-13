'''
variants.py

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

import pysam
import numpy as np
import cv2
from PIL import Image
import itertools

from denovonet.settings import OVERHANG, IMAGE_WIDTH, PLACEHOLDER_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, NUCLEOTIDES
from denovonet.encoders import baseEncoder, VariantClassValue

class SingleVariant():
    """
    Class for encoding a variant and the area around as 2 numpy arrays:
    numpy array with nucleotides encoded and 
    numpy array with corresponding base quality encoded
    """
    
    def __init__(self, chromosome, start, end, bam_path, REREFERENCE_GENOME):
        # variant location
        self.chromosome = chromosome
        self.start = int(start)
        self.end = int(end)
        
        # reference genome 
        self.REFERENCE_GENOME = REREFERENCE_GENOME

        # bam file
        self.bam_path = bam_path

        # encoded nucleotides pileup placeholder
        self.pileup_encoded = np.zeros((IMAGE_HEIGHT, PLACEHOLDER_WIDTH)).astype(int)

        # encoded qualities placeholder
        self.quality_encoded = np.zeros((IMAGE_HEIGHT, PLACEHOLDER_WIDTH)).astype(int)

        # run encoding the variant as 2 numpy arrays
        self.encode_pileup()
        
    # variant area: variant location +- N positions
    @property
    def region_start(self):
        return self.start - OVERHANG - 2
    
    @property
    def region_end(self):
        return self.start + OVERHANG - 2
    
    # reference sequence in variant location +- N positions
    @property
    def region_reference_sequence(self):
        return self.REFERENCE_GENOME.fetch(self.chromosome, self.region_start+1, self.region_end+2)
    
    # variant location
    @property
    def target_range(self):
        return range(self.start-1, self.end-1)

    # loaded bam file
    @property
    def bam_data(self):
        return pysam.AlignmentFile(self.bam_path, "rb")
    
    # calculate coverage at variant location
    @property
    def start_coverage(self):
        start_coverage_arrays = self.bam_data.count_coverage(self.chromosome, self.start-1, self.start)
        return sum([coverage[0] for coverage in start_coverage_arrays])
    
    def encode_pileup(self):
        """
            Iterates over all the reads in the area of interest and
            encodes every read as 2 numpy arrays: 
            encoded nucleotides and corresponding qualities
        """
        for idx, read in enumerate(self.bam_data.fetch(reference=self.chromosome, start=self.start, end=self.end)):
            if idx >= IMAGE_HEIGHT:
                break
            self.pileup_encoded[idx, :], self.quality_encoded[idx, :] = (
                self._get_read_encoding(read, False)
            )
            
    def _get_read_encoding(self, read, debug=False):
        """
            Calculates nucleotide encoding and qualities numpy arrays for one read
        """
        
        self._read = read
        # read properties
        self._cigar = self._read.cigar
        self._seq = self._read.seq
        self._query_qualities = np.array(self._read.query_qualities).astype(int)
        self._mapq = self._read.mapq

        # setting initial zeros for pileup and quality
        pileup = np.zeros((PLACEHOLDER_WIDTH, ))
        quality = np.zeros((PLACEHOLDER_WIDTH, ))

        # get reference genome for read
        self._ref = self.REFERENCE_GENOME.fetch(self.chromosome, self._read.reference_start, self._read.reference_start + 2*len(read.seq))
        
        # offset if reference_start before interest area [start - OVERHANG - 1, start + OVERHANG -1]
        offset = max(0, (self.start - OVERHANG - 1) - read.reference_start)

        #offset if reference_start inside interest area [start - OVERHANG - 1, start + OVERHANG -1]
        offset_picture = max(0, read.reference_start - (self.start - OVERHANG - 1))

        # pointers to reference genome positions
        genome_start_position = 0 + offset
        genome_end_position = 0

        # pointers to bases in read position
        base_start_position = self._calculate_base_start_position(genome_start_position, genome_end_position)
        base_end_position = 0

        # pointers to picture position
        picture_start_position = 0 + offset_picture
        picture_end_position = 0 + offset_picture
        
        # skip bad reads
        if not self._cigar or self._cigar[0][0] in (4, 5) or offset_picture > PLACEHOLDER_WIDTH:
            return pileup, quality

        #iterate over all cigar pairs
        for iter_num, (cigar_value, cigar_num) in enumerate(self._cigar):
            
            # update pointers end position
            base_end_position, genome_end_position = self._update_positions(
                cigar_value, cigar_num, base_end_position, genome_end_position
            )

            #we don't reach interest region
            if genome_end_position < genome_start_position:
                if base_start_position < base_end_position:
                    base_start_position = base_end_position
                continue
            
            # correction if we outside interest region
            genome_end_position = min(
                genome_end_position, 
                genome_start_position + PLACEHOLDER_WIDTH - picture_end_position
            )

            base_end_position = min(
                base_end_position, 
                base_start_position + PLACEHOLDER_WIDTH - picture_end_position
            )
            
            picture_step = min(
                cigar_num, 
                max(
                    genome_end_position - genome_start_position, 
                    base_end_position - base_start_position
                ))

            picture_end_position += picture_step

            # calculate quality
            quality[picture_start_position:picture_end_position] = self._calculate_quality(
                cigar_value, base_start_position, base_end_position, 
                genome_start_position, genome_end_position, picture_step
            )

            # calculate pilup
            pileup[picture_start_position:picture_end_position] = self._calculate_pileup(
                cigar_value, base_start_position, base_end_position, picture_step
            )

            # move pointers
            base_start_position = base_end_position
            genome_start_position = genome_end_position
            picture_start_position = picture_end_position

            if picture_end_position >= PLACEHOLDER_WIDTH:
                break

        return (pileup, quality)
    

    def _calculate_base_start_position(self, genome_start_position, genome_end_position):
        """
            calculates base_start_position if genome_start_position > read.reference_start
        """
        if genome_start_position <= genome_end_position:
            return 0

        cigar_line = [cigar_value for cigar_value, cigar_num in self._cigar for x in range(cigar_num)]
        
        base_start_position = 0

        for cigar_value in cigar_line:
            base_start_position, genome_end_position = self._update_positions(
                cigar_value, 1, base_start_position, genome_end_position
            )

            if genome_end_position >= genome_start_position:
                break
        
        return base_start_position
    
    def _update_positions(self, cigar_value, cigar_num, base_position, genome_position):
        """
            updates current positions based on cigar_value
        """
        # match
        if cigar_value == 0:
            base_position += cigar_num
            genome_position += cigar_num
        # insertion
        elif cigar_value == 1:
            base_position += cigar_num
        # deletion
        elif cigar_value == 2:
            genome_position += cigar_num
        elif cigar_value == 4:
            base_position += cigar_num
            genome_position += cigar_num
        elif cigar_value == 5:
            base_position += cigar_num
        else:
            raise ValueError('Unsupported cigar value: {}'.format(cigar_value))

        return base_position, genome_position

    def _calculate_quality(self, cigar_value, 
                    base_start_position, base_end_position, 
                    genome_start_position, genome_end_position, picture_step):
        """
            calculates quality array values 
            quality value is calculated as multiplication of 
            base quality and mapping quality divided by 10
            
            if the quality is outside region of interest or 
            doesn't correspont to a varint it's also
            divided by 3
        """

        read = self._read
        start = self.start - 1 
        end = self.end - 1 
        query_qualities = self._query_qualities
        mapq = self._mapq
        ref = self._ref
        seq = self._seq
        
        absolute_genome_start = read.reference_start + genome_start_position
        absolute_genome_end = read.reference_start + genome_end_position
        current_genome_range = np.arange(absolute_genome_start, absolute_genome_end)

        current_quality = np.zeros_like((picture_step, ))

        # match
        if cigar_value == 0:
            current_quality = query_qualities[base_start_position:base_end_position] * mapq // 10
            matching_mask = (
                np.array(list(seq[base_start_position:base_end_position])) == 
                np.array(list(ref[genome_start_position:genome_end_position]))
            )
            non_interest_region = (current_genome_range < start) | (current_genome_range >= end)
            current_quality[matching_mask & non_interest_region] //= 3
        
        #insertion
        elif cigar_value == 1:
            current_quality = query_qualities[base_start_position:base_end_position] * mapq // 10

        #deletion
        elif cigar_value == 2:
            current_quality = np.ones((picture_step, ))*query_qualities[base_end_position] * mapq // 10
        
        return current_quality

    def _calculate_pileup(self, cigar_value, base_start_position, base_end_position, picture_step):
        """
            calculates pileup array values 
            nucleotide is encoded as corresponding value of baseEncoder
            
        """
        current_pileup = np.zeros_like((picture_step, ))

        # match and insertion
        if cigar_value in (0, 1):
            #encode bases
            sub_seq = self._seq[base_start_position:base_end_position]
            current_pileup = self._get_encodings(cigar_value, sub_seq)
        
        #deletion 
        elif cigar_value == 2:
            #encode bases
            sub_seq = [-1] * picture_step
            current_pileup = self._get_encodings(cigar_value, sub_seq)
        
        return current_pileup

    def _get_encodings(self, cigar_value, bases):
        """
            calculates pileup array values 
            nucleotide is encoded as corresponding value of baseEncoder
            
        """
        encoding_match = {
            'A': baseEncoder.A,
            'C': baseEncoder.C,
            'T': baseEncoder.T,
            'G': baseEncoder.G,
            'N': baseEncoder.EMPTY,
        }
        encoding_insertion = {
            'A': baseEncoder.IN_A,
            'C': baseEncoder.IN_C,
            'T': baseEncoder.IN_T,
            'G': baseEncoder.IN_G,
            'N': baseEncoder.IN_A,
        }

        result = np.zeros((len(bases), ))

        if cigar_value == 2:
            return result + baseEncoder.DEL

        for idx, base in enumerate(bases):
            if cigar_value == 0:
                result[idx] = encoding_match.get(base, 0)
            
            if cigar_value == 1:
                result[idx] = encoding_insertion.get(base, 0)
        
        return result

class TrioVariant():
    """
        Class for merging 3 objects of SingleVariant class
        for the trio as RGB image
    """
    
    def __init__(self, child_variant, father_variant, mother_variant):
        # SingleVariant objects for a trio
        self.child_variant = child_variant
        self.father_variant = father_variant
        self.mother_variant = mother_variant

        # Create singleton variant images
        self.child_variant_image = self.create_singleton_variant_image(self.child_variant.pileup_encoded, self.child_variant.quality_encoded)
        self.father_variant_image = self.create_singleton_variant_image(self.father_variant.pileup_encoded, self.father_variant.quality_encoded)
        self.mother_variant_image = self.create_singleton_variant_image(self.mother_variant.pileup_encoded, self.mother_variant.quality_encoded)

        # Combine singleton images
        self.image = self.create_trio_variant_image()

    def create_singleton_variant_image(self, variant_pileup, variant_quality):
        """
            Combines encoded nucleotides and bases quality arrays from SingleVariant
            as one array where every nucleotied is one-hot encoded with the value
            that equal corresponding quality value. 
        """
        variant_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

        for row_index, row in enumerate(variant_pileup):
            for column_index, value in enumerate(row):
                pileup_coordinates = (row_index, column_index)

                base = variant_pileup[pileup_coordinates]
                pixel_value = variant_quality[pileup_coordinates]

                if base == baseEncoder.A or base == baseEncoder.IN_A:
                    variant_image[row_index, column_index * 4 + 0] = pixel_value
                elif base == baseEncoder.C or base == baseEncoder.IN_C:
                    variant_image[row_index, column_index * 4 + 1] = pixel_value
                elif base == baseEncoder.T or base == baseEncoder.IN_T:
                    variant_image[row_index, column_index * 4 + 2] = pixel_value
                elif base == baseEncoder.G or base == baseEncoder.IN_G:
                    variant_image[row_index, column_index * 4 + 3] = pixel_value
                elif base == baseEncoder.DEL:
                    variant_image[row_index, column_index*4:column_index*4+4] = pixel_value

        return variant_image

    def normalize_image(self,image):
        """
            Normalize pixel values to be in [0, 1]
        """
        image = image.astype(float)
        image /= 255
        
        return image

    def create_trio_variant_image(self):
        """
            Combines trio arrays as RGB image.
        """
        image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        image[:,:,0] = self.child_variant_image
        image[:,:,1] = self.father_variant_image
        image[:,:,2] = self.mother_variant_image

        return image

    def predict(self, model):
        """
            Applies the model to RGB image
            and gets DNM prediction
        """
        expanded_image = np.expand_dims(self.image, axis=0)
        normalized_image = expanded_image.astype(float) / 255
        prediction = model.predict(preprocess_image(normalized_image))
        return prediction

    @staticmethod
    def display_image(image):
        """
            Displays RGB image
        """
            
        cv2.imwrite('', image) 

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image_path, image):
        """
            Saves image to image_path
        """
        cv2.imwrite(image_path, image) 

    @staticmethod
    def predict_image_path(image_path, model):
        """
            Applies the model to RGB image within the image_path
            and gets DNM prediction 
        """
        image = Image.open(image_path)
        normalized_image = np.array(image).astype(float) / 255
        expanded_image = np.expand_dims(normalized_image, axis=0)
        prediction = model.predict(preprocess_image(expanded_image))
        return prediction


def preprocess_image(img):
    """
        Preprocess RGB image before passing to the network
    """
    if IMAGE_CHANNELS == 2:
        if (len(img.shape) == 3) and (img.shape[2] == 3):
            n1, n2, n3 = img.shape
            img_new = np.zeros(shape=(n1, n2, n3 - 1))
            
            img_new[:, :, 0] = img[:, :, 0].copy()
            img_new[:, :, 1] = np.maximum(img[:, :, 1], img[:, :, 2]).copy()

            return img_new

        elif (len(img.shape) == 4)  and (img.shape[3] == 3):
            n1, n2, n3, n4 = img.shape
            img_new = np.zeros(shape=(n1, n2, n3, n4 - 1))
            
            img_new[:, :, :, 0] = img[:, :, :, 0].copy()
            img_new[:, :, :, 1] = np.maximum(img[:, :, :, 1], img[:, :, :, 2]).copy()

            return img_new

        else:
            raise Exception("Shape of the image is incorrect:", img.shape)

    return img