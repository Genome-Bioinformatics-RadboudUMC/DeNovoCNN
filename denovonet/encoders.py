'''
encoders.py

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

class baseEncoder:
    EMPTY = 0
    A = 1
    C = 2
    T = 3
    G = 4
    IN_A = 5
    IN_C = 6
    IN_T = 7
    IN_G = 8
    DEL = 9

class VariantClassValue:
    snp = 0
    deletion = 1
    insertion = 2
    unknown = 3
    
class VariantInheritance:
    DNM = 0
    IV = 1
