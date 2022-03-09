'''
settings.py

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

# IMAGE PARAMETERS
OVERHANG = 20 #number of nucleotides to each side of the center
NUCLEOTIDES = 4

IMAGE_CHANNELS = 3 # 3 or 2 : (child, father, mother) or (child, max(father, mother))
IMAGE_WIDTH = 4 * (2 * OVERHANG + 1)
IMAGE_HEIGHT = 160 #Pileup height

CHANNEL_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
SINGLETON_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

PLACEHOLDER_WIDTH = 2 * OVERHANG + 1

IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

MINIMAL_COVERAGE = 7

BATCH_SIZE = 32

VARIANT_CLASSES = ['DNM','IV']    
NUMBER_CLASSES = len(VARIANT_CLASSES)

MODEL_ARCHITECTURE = 'advanced_cnn_binary' # cnn 
CLASS_MODE = 'binary' #categorical
