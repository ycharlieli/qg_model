import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.ndimage import map_coordinates
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

class QGCDA:
    def __init__(self,m=None,m_ref=None,interpolant='linear',Nobs=256,dTobs=0.1,mu=0.1):
        self.m = m
        self.m_ref = m_ref
        self.interpolant =  interpolant
        self.Nobs = Nobs
        self.dTobs = dTobs
        self.step_model = int(self.m.Nx/self.Nobs)
        self.step_ref = int(self.m_ref.Nx/self.Nobs)
        self.dt = np.min([self.m.dt,self.m_ref.dt])
        self.intvl_model = int(round(self.m.dt/self.dt)) # check step instead time to avoid trunction error
        self.intvl_ref = int(round(self.m_ref.dt/self.dt))
        self.intvl_cda = int(round(self.dTobs/self.dt))
        self.mu = mu
        self._init_grid()

    def _init_grid(self):
        self.x_obs = cp.linspace(0,self.m.Lx,self.Nobs) # position of observation
        self.y_obs = cp.linspace(0,self.m.Ly,self.Nobs)
        x_model, y_model = cp.meshgrid(cp.arange(self.m.Nx),cp.arange(self.m.Ny))
        ratio = self.Nobs/self.m.Nx
        rcoord_x =  x_model.ravel() * ratio# relative coordinate of model to observation
        rcoord_y =  y_model.ravel() * ratio
        self.rcoord_model = cp.array([rcoord_x,rcoord_y])
        

    def _step_cda(self):
        q_m_r = ifft2(self.m.q_hat).real # back to physical space (real space)
        q_ref_r = ifft2(self.m_ref.q_hat).real

        q_m_sub = q_m_r[::self.step_model,::self.step_model]
        q_ref_sub = q_ref_r[::self.step_ref,::self.step_ref]

        q_m_sub = q_m_sub - cp.mean(q_m_sub) #zero correction for periodic domain
        q_ref_sub = q_ref_sub - cp.mean(q_ref_sub)
        if self.interpolant == 'linear':
            self.cda_forcing = self.mu*(self._linear_intp(q_ref_sub)-self._linear_intp(q_m_sub))
        elif self.interpolant == 'block':
            self.cda_forcing = self.mu*(self._block_intp(q_ref_sub)-self._block_intp(q_m_sub))

        self.m.cda_term = fft2(self.cda_forcing)# back to fourier space


    def _linear_intp(self,phi):
        # using map_coordinate to deal with periodic  boundary
        phi_intp = map_coordinates(phi,self.rcoord_model,order=1,mode='wrap',prefilter=False)
        return phi_intp.reshape(self.m.Nx,self.m.Ny)

    def _block_intp(self,phi):
        # using map_coordinate to deal with periodic  boundary
        phi_intp = map_coordinates(phi,self.rcoord_model,order=0,mode='wrap',prefilter=False)
        return phi_intp.reshape(self.m.Nx,self.m.Ny)

    # def _block_intp(self):
    #     pass
    def create_nc(self,nf):
        self.m.create_nc(nf)
        self.cdaF_var = self.m.ds.createVariable('cdaF', 'f8', ('time', 'x', 'y'), zlib=False)
        self.tecdak_var = self.m.ds.createVariable('tecdak', 'f8', ('time', 'k'), zlib=False)
        self.tzcdak_var = self.m.ds.createVariable('tzcdak', 'f8', ('time', 'k'), zlib=False)
        self.fecdak_var = self.m.ds.createVariable('fecdak', 'f8', ('time', 'k'), zlib=False)
        self.fzcdak_var = self.m.ds.createVariable('fzcdak', 'f8', ('time', 'k'), zlib=False)

    def save_var(self,it):
        self.m.save_var(it)
        # cda term
        self.cdaF_var[it,:,:] = ifft2(self.m.cda_term).real.get()
        self.tecdak_var[it,:], self.tzcdak_var[it,:],self.fecdak_var[it,:], self.fzcdak_var[it,:] = self.m.get_diagCda(self.m.p_hat,self.m.q_hat,self.m.cda_term)
        self.m.ds.sync()
    # def save_ref(self,ds):



    def cda_run(self,scheme='ab3',trst=0,tmax=40,tsave=200,nsave=100,savedir='run_cda0',saveplot=False):
        
        self.m.t = trst
        self.m_ref.t = trst    
        self.m.n_steps = 0
        self.m_ref.n_steps = 0

        self.m.savedir = savedir
        os.makedirs(self.m.savedir, exist_ok=True)

        total_steps = int(round(tmax/self.dt))

        print(f"Starting CDA. dTobs={self.dTobs}, tmax={tmax}")
        print(f"Ratios -> Model: {self.intvl_model}, Ref: {self.intvl_ref}, Obs: {self.intvl_cda}")
        insave=nsave
        nf=0

        for n in range(total_steps+1):
            if n % self.intvl_cda == 0:
                self._step_cda()
            
            
            
            if n % self.intvl_ref ==0:
                self.m_ref._step_forward(scheme=scheme)
                self.m_ref.t += self.m_ref.dt
                self.m_ref.n_steps +=1

            if n % self.intvl_model ==0:
                if self.m.n_steps % tsave == 0:

                    if insave == nsave:
                        itsave =0 #time index
                        if nf > 0 : self.m.ds.close()
                        self.create_nc(nf)
                        insave=0 #save number index
                        nf+=1
                
                    self.save_var(itsave)
                    E_crt = self.m.get_Etot(self.m.p_hat)/self.m.Nx/self.m.Ny
                    Vrms_crt = self.m.get_Vrms(self.m.p_hat)
                    print(f"   step {self.m.n_steps:7d}  t={self.m.t:9.6f}s E={E_crt:.4e} Vrms={Vrms_crt:.4e}", end="\n")
                    if saveplot:
                        self.m.plot_diag()
                    itsave +=1
                    insave +=1  
                self.m._step_forward(scheme=scheme)
                self.m.t += self.m.dt
                self.m.n_steps += 1

        self.m.ds.close()
        
        print('Done.')