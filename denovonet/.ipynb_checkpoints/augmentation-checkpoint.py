'''
augmentation.py

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

from denovonet.settings import NUCLEOTIDES
import numpy as np


class CustomAugmentation(object):
    """ Defines a custom augmentation class. Randomly applies one of transformations."""

    def __init__(self, probability=0.9, reads_cropping=False, reads_shuffling=False, multi_nucleotide_snp=False,
                 nucleotides_relabeling=False, channels_switching=False, seed=None):
        self.probability = probability
        self.reads_cropping = reads_cropping
        self.reads_shuffling = reads_shuffling
        self.multi_nucleotide_snp = multi_nucleotide_snp
        self.nucleotides_relabeling = nucleotides_relabeling
        self.channels_switching = channels_switching
        self.transformations = []

        if seed:
            np.random.seed(seed)

    def _check_augmentations(self):
        """
            Creates augmentations list.
        """

        self.transformations = []

        if self.reads_cropping:
            self.transformations.append(self._reads_cropping)
        if self.reads_shuffling:
            self.transformations.append(self._reads_shuffling)
        if self.multi_nucleotide_snp:
            self.transformations.append(self._multi_nucleotide_snp)
        if self.nucleotides_relabeling:
            self.transformations.append(self._nucleotides_relabeling)
        if self.channels_switching:
            self.transformations.append(self._channels_switching)

    @staticmethod
    def _reads_cropping(img):
        """
            Returns image with randomly cropped reads.
        """
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))

        nreads_c = max(5, nreads_c)
        nreads_f = max(5, nreads_f)
        nreads_m = max(5, nreads_m)

        nreads_c = np.random.choice(np.arange(5, nreads_c + 1))
        nreads_f = np.random.choice(np.arange(5, nreads_f + 1))
        nreads_m = np.random.choice(np.arange(5, nreads_m + 1))

        new_img[nreads_c:, :, 0] = 0.
        new_img[nreads_f:, :, 1] = 0.
        new_img[nreads_m:, :, 2] = 0.

        return new_img

    @staticmethod
    def _nucleotides_relabeling(img):
        """
            Returns image with nucleotides relabeled (swapped),
            for example A->T, T->C, C->G, G->A.
        """
        new_img = img.copy()

        new_ordering = list(range(NUCLEOTIDES))
        np.random.shuffle(new_ordering)

        for old_idx, new_idx in enumerate(new_ordering):
            new_img[:, old_idx::NUCLEOTIDES, :] = img[:, new_idx::NUCLEOTIDES, :].copy()
        return new_img

    @staticmethod
    def _multi_nucleotide_snp(img):
        """
            Returns image with added SNP to the left or
            to the right of original SNP.
        """

        def differ_array(arr):
            """
                Randomly changes original array
            """

            for_mask = np.sum(np.sum(arr > 0, axis=2), axis=0)

            if np.sum(for_mask > 0) == 0:
                return arr

            min_val = np.min(for_mask[for_mask > 0])

            rand_arr = np.random.randint(-8, high=8, size=arr.shape)
            arr[arr > 0] += rand_arr[arr > 0]

            arr[:, for_mask > min_val, :] //= 3

            arr = np.clip(arr, 0, 255)
            return arr

        def get_rearranged_snp(snp, n=1):
            """
                Applies random nucleotides relabeling to snp to get different snp.
            """
            image_new = np.tile(snp, (1, n, 1))

            for i in range(n):
                image_new[:, i * NUCLEOTIDES:(i + 1) * NUCLEOTIDES, :] = CustomAugmentation._nucleotides_relabeling(
                    image_new[:, i * NUCLEOTIDES:(i + 1) * NUCLEOTIDES, :])

            return image_new

        def insert_snp_left(image, n=1):
            """
                Insertion of the new SNP to the left.
            """
            image_new = image.copy()
            image_new[:, :(20 - n) * NUCLEOTIDES, :] = image_new[:, n * NUCLEOTIDES:20 * NUCLEOTIDES, :].copy()
            image_new[:, (20 - n) * NUCLEOTIDES: 20 * NUCLEOTIDES, :] = differ_array(
                get_rearranged_snp(image_new[:, 20 * NUCLEOTIDES: 21 * NUCLEOTIDES, :], n))
            return image_new

        def insert_snp_right(image, n=1):
            """
                Insertion of the new SNP to the right.
            """
            return insert_snp_left(image[:, ::-1, :], n)[:, ::-1, :]

        func = np.random.choice([insert_snp_left, insert_snp_right])
        n = np.random.choice(range(1, 3))

        return func(img, n)

    @staticmethod
    def _reads_shuffling(img):
        """
            Shuffling the reads order.
        """
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))

        np.random.shuffle(new_img[:nreads_c, :, 0])
        np.random.shuffle(new_img[:nreads_f, :, 1])
        np.random.shuffle(new_img[:nreads_m, :, 2])

        return new_img

    @staticmethod
    def _channels_switching(img):
        """
            Switching parental channels.
        """
        new_img = img.copy()
        new_img[:, :, 1], new_img[:, :, 2] = new_img[:, :, 2].copy(), new_img[:, :, 1].copy()

        return new_img

    def __call__(self, img):
        """
            Applies random augmentation from the list.
        """

        if img.shape[2] != 3:
            print(img.shape)
            raise Exception("Wrong image format!")

        random_number = np.random.random()

        if random_number > self.probability:
            pass
        else:
            self._check_augmentations()
            transformation = np.random.choice(self.transformations)
            return transformation(img)

        return img
    
class CustomAugmentationLRS (CustomAugmentation):
    """Defines an augmentation class specifically customized for LRS-encoded images"""
    def __init__(self, probability=0.9, reads_cropping=False, reads_shuffling=False, multi_nucleotide_snp=False, 
                 nucleotides_relabeling=False, channels_switching=False, seed=None):
        super().__init__(probability, reads_cropping, reads_shuffling, multi_nucleotide_snp, 
                         nucleotides_relabeling, channels_switching, seed)
    
    @staticmethod
    def _reads_cropping(img):
        """
        Crops reads randomly from different parts of the image (haplotypes)
        """
        new_img = img.copy()

        nreads_c_hp0, nreads_f_hp0, nreads_m_hp0 = tuple(np.sum(np.sum(new_img[:52, :, :], axis=1) > 0., axis=0))
        nreads_c_hp2, nreads_f_hp2, nreads_m_hp2 = tuple(np.sum(np.sum(new_img[53:105, :, :], axis=1) > 0., axis=0))
        nreads_c_hp1, nreads_f_hp1, nreads_m_hp1 = tuple(np.sum(np.sum(new_img[106:, :, :], axis=1) > 0., axis=0))
        
        not_suitable = ((nreads_c_hp0 > 53) or (nreads_c_hp2 > 53) or (nreads_c_hp1 > 53) or 
        (nreads_f_hp0 > 53) or (nreads_f_hp2 > 53) or (nreads_f_hp1 > 53) or 
        (nreads_m_hp0 > 53) or (nreads_m_hp2 > 53) or (nreads_m_hp1 > 53))
        
        # if one of the channels has too many reads for one of the haplotypes, no augmentation
        if not_suitable:
            return new_img

        nreads_c_hp0, nreads_f_hp0, nreads_m_hp0 = (max(3, nreads_c_hp0), max(3, nreads_f_hp0), max(3, nreads_m_hp0))
        nreads_c_hp1, nreads_f_hp1, nreads_m_hp1 = (max(3, nreads_c_hp1), max(3, nreads_f_hp1), max(3, nreads_m_hp1))
        nreads_c_hp2, nreads_f_hp2, nreads_m_hp2 = (max(3, nreads_c_hp2), max(3, nreads_f_hp2), max(3, nreads_m_hp2))
        
        nreads_c_hp0 = np.random.choice(np.arange(3, nreads_c_hp0 + 1))
        nreads_f_hp0 = np.random.choice(np.arange(3, nreads_f_hp0 + 1))
        nreads_m_hp0 = np.random.choice(np.arange(3, nreads_m_hp0 + 1))
        
        nreads_c_hp1 = np.random.choice(np.arange(3, nreads_c_hp1 + 1))
        nreads_f_hp1 = np.random.choice(np.arange(3, nreads_f_hp1 + 1))
        nreads_m_hp1 = np.random.choice(np.arange(3, nreads_m_hp1 + 1))
        
        nreads_c_hp2 = np.random.choice(np.arange(3, nreads_c_hp2 + 1))
        nreads_f_hp2 = np.random.choice(np.arange(3, nreads_f_hp2 + 1))
        nreads_m_hp2 = np.random.choice(np.arange(3, nreads_m_hp2 + 1))
        
        # haplotype 0:
        new_img[nreads_c_hp0:52, :, 0] = 0.
        new_img[nreads_f_hp0:52, :, 1] = 0.
        new_img[nreads_m_hp0:52, :, 2] = 0.
        
        # haplotype 2:
        new_img[(53 + nreads_c_hp2):105, :, 0] = 0.
        new_img[(53 + nreads_f_hp2):105, :, 1] = 0.
        new_img[(53 + nreads_m_hp2):105, :, 2] = 0.
        
        # haplotype 1:
        new_img[(106 + nreads_c_hp1):, :, 0] = 0.
        new_img[(106 + nreads_f_hp1):, :, 1] = 0.
        new_img[(106 + nreads_m_hp1):, :, 2] = 0.
        
        return new_img
    
    @staticmethod
    def _reads_shuffling (img):
        """
        Shuffles the reads from the same haplotype randomly
        """
        new_img = img.copy()

        nreads_c_hp0, nreads_f_hp0, nreads_m_hp0 = tuple(np.sum(np.sum(new_img[:53, :, :], axis=1) > 0., axis=0))
        nreads_c_hp2, nreads_f_hp2, nreads_m_hp2 = tuple(np.sum(np.sum(new_img[53:106, :, :], axis=1) > 0., axis=0))
        nreads_c_hp1, nreads_f_hp1, nreads_m_hp1 = tuple(np.sum(np.sum(new_img[106:, :, :], axis=1) > 0., axis=0))
        
        not_suitable = ((nreads_c_hp0 > 53) or (nreads_c_hp2 > 53) or (nreads_c_hp1 > 53) or 
        (nreads_f_hp0 > 53) or (nreads_f_hp2 > 53) or (nreads_f_hp1 > 53) or 
        (nreads_m_hp0 > 53) or (nreads_m_hp2 > 53) or (nreads_m_hp1 > 53))
        
        # if one of the channels has too many reads for one of the haplotypes, no augmentation
        if not_suitable:
            return new_img
        
        # haplotype 0:
        if (nreads_c_hp0 > 1) and (nreads_f_hp0 > 1) and (nreads_m_hp0 > 1):
            np.random.shuffle(new_img[:nreads_c_hp0, :, 0])
            np.random.shuffle(new_img[:nreads_f_hp0, :, 1])
            np.random.shuffle(new_img[:nreads_m_hp0, :, 2])
        
        # haplotype 2:
        if (nreads_c_hp2 > 1) and (nreads_f_hp2 > 1) and (nreads_m_hp2 > 1):
            np.random.shuffle(new_img[53:(53 + nreads_c_hp2), :, 0])
            np.random.shuffle(new_img[53:(53 + nreads_f_hp2), :, 1])
            np.random.shuffle(new_img[53:(53 + nreads_m_hp2), :, 2])
        
        # haplotype 1:
        if (nreads_c_hp1 > 1) and (nreads_f_hp1 > 1) and (nreads_m_hp1 > 1):
            np.random.shuffle(new_img[106:(106 + nreads_c_hp1), :, 0])
            np.random.shuffle(new_img[106:(106 + nreads_f_hp1), :, 1])
            np.random.shuffle(new_img[106:(106 + nreads_m_hp1), :, 2])
        
        return new_img
            
