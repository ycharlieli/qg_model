import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.interpolate import interpn
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

class QGCDA(QGModel):
    def __init__(self,m=None,m_ref=None,interpolant='linear',Nobs=256,dTobs=0.1,mu=0.1):
        self.m = m
        self.m_ref = m_ref
        self.interpolant = 'linear'
        self.Nobs = Nobs
        self.dTobs = dTobs
        self.dt = np.min([self.m.dt,self.m_ref.dt])
        self.step_model = self.m.Nx/self.Nobs
        self.step_ref = self.m_ref.Nx/self.Nobs
        self.mu = mu
        self._init_grid()

    def _init_grid(self):
        self.x_obs = cp.linspace(0,self.m.Lx,self.Nobs) # position of observation
        self.y_obs = cp.linspace(0,self.m.Ly,self.Nobs)
        self.points_m = cp.c_[                      # position of model to be corrected
            self.m.x2d.ravel()[:,cp.newaxis],
            self.m.y2d.ravel()[:,cp.newaxis]
        ]

    def _step_cda(self,interpolant='linear'):
        q_r = ifft2(self.m.q_hat).real # back to physical space 
        q_r_ref = ifft2(self.m_ref.q_hat).real

        q_sub = q_r[::self.step_model,::self.step_model]
        q_ref_sub = q_r_ref[::self.step_ref,::self.step_ref]

        q_sub = q_sub - cp.mean(q_sub) #zero correction for periodic domain
        q_ref_sub = q_ref_sub - cp.mean(q_ref_sub)
        if interpolant == 'linear':
            self.m.cda_term = fft2(self.mu*(self._linear_intp(q_ref_sub)-self._linear_intp(q_sub)))# back to fourier space


    def _linear_intp(self,phi):
        phi_intp = (interpn((self.x_obs,self.y_obs),phi,self.points_m)).reshape(self.m.x2d.shape)
        return phi_intp



    # def _block_intp(self):
    #     pass
    
    # def save_cda(self):

    # def save_ref(self):


    def cda_run(self,scheme='ab3',tmax=40,tsave=200,nsave=100,tplot=1000,savedir='run_cda0',saveplot=False):
        self.tmax = tmax
        self.tsave = tsave
        self.m.t = 0    
        self.m.n_steps = 0
        self.m_ref.n_steps = 0
        self.m.savedir = savedir
        os.makedirs(self.m.savedir, exist_ok=True)
        insave=nsave
        inplot = 0
        nf=0

        for n in range(int(tmax/self.dt)+1):
            
            if np.isclose(np.mod(self.m.t,self.dTobs),0):
                self._step_cda(interpolant=self.interpolant)
                
            if np.isclose(np.mod(self.m.t,self.m.dt),0):
                if insave == nsave:
                    itsave =0 #time index
                    if nf > 0 : self.m.ds.close()
                    self.m.create_nc(nf)
                    insave=0 #save number index
                    nf+=1
                if n%tsave == 0:
                    self.m.save_var(itsave)
                    print(f"step {n:7d}  t = {self.m.t:9.6f} s  E = {self.m.get_Etot(self.m.p_hat)/self.m.Nx/self.m.Ny:.4e}", end="\n")
                    if saveplot:
                        self.m.plot_diag()
                    itsave +=1
                    insave +=1
                if n%tplot ==0:
                    print(f"\t saving figure", end="\n")
                    self.m.save_snapshot(inplot) 
                    inplot+=1
                self.m._step_forward(scheme=scheme)
                self.m.n_steps+=1

            if np.isclose(np.mod(self.m.t,self.m_ref.dt),0):
                self.m_ref._step_forward(scheme=scheme)
                self.m_ref.n_steps +=1

            self.m.t += self.dt
        self.ds.close()
        print('Done.')