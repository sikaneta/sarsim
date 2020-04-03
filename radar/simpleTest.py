#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:34:11 2020

@author: ishuwa
"""


from numba import cuda
if cuda.is_available():
    from antenna.pattern import antennaResponseCudaMem as antennaResp
else:
    from antenna.pattern import antennaResponseMultiCPU as antennaResp
    
print("Hello")