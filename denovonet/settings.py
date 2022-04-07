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

# IMAGE PARAMETERS
OVERHANG = 20 # number of nucleotides to each side of the center
PLACEHOLDER_WIDTH = 2 * OVERHANG + 1

NUCLEOTIDES = 4

IMAGE_CHANNELS = 3 #RGB image
IMAGE_WIDTH = NUCLEOTIDES * (2 * OVERHANG + 1)
IMAGE_HEIGHT = 160 # pileup height (= image height)

