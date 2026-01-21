import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy as sp
from scipy.fft import fft2,rfft2,ifft2,fftshift,irfft2
import numpy_groupies as npg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import set_global_backend
set_global_backend(cufft)
import gc
import netCDF4 as nc
import os
from turb2d import QGModel

class CDAQG(QGModel):
    def __init__(self,m=None,m_ref=None):

    def coarsing(self,k_cut=16):
        # the k_cut is in 2pi domain (2pi*k for the other domain)