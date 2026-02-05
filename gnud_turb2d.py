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

class QGGNUD:
    """Grid Nudging: Apply nudging only on observation grid points, zero elsewhere"""
    def __init__(self,m=None,m_ref=None,interpolant='linear',Nobs=256,dTobs=0.1,mu=0.1):
        """Initialize Grid Nudging with model and reference models
        
        Args:
            m: Model to assimilate (nudged model)
            m_ref: Reference/truth model
            interpolant: 'linear' or 'block' interpolation method
            Nobs: Number of observation grid points (Nobs x Nobs)
            dTobs: Time interval between observations
            mu: Nudging strength parameter
        """
        self.m = m
        self.m_ref = m_ref
        self.interpolant = interpolant
        self.Nobs = Nobs
        self.dTobs = dTobs
        self.dt = np.min([self.m.dt, self.m_ref.dt])
        self.intvl_model = int(round(self.m.dt/self.dt)) # check step instead time to avoid trunction error
        self.intvl_ref = int(round(self.m_ref.dt/self.dt))
        self.intvl_gnud = int(round(self.dTobs/self.dt))
        self.mu = mu
        self._init_grid()

    def _init_grid(self):
        """Initialize observation grid and coordinate mappings"""
        self.x_obs = cp.linspace(0, self.m.Lx, self.Nobs) # position of observation
        self.y_obs = cp.linspace(0, self.m.Ly, self.Nobs)
        x_model_idx, y_model_idx = cp.meshgrid(cp.arange(self.m.Nx), cp.arange(self.m.Ny))
        ratio = self.Nobs / self.m.Nx
        rcoord_x = x_model_idx.ravel() * ratio # relative coordinate of model to observation
        rcoord_y = y_model_idx.ravel() * ratio
        self.rcoord_model = cp.array([rcoord_y, rcoord_x])
        x_obs_idx, y_obs_idx = cp.meshgrid(cp.arange(self.Nobs), cp.arange(self.Nobs))
        
        scale_m = self.m.Nx / self.Nobs
        ds_x_m = x_obs_idx * scale_m # Coordinates for Model -> Obs
        ds_y_m = y_obs_idx * scale_m
        self.coords_ds_m = cp.array([ds_y_m, ds_x_m])
        
        
        scale_ref = self.m_ref.Nx / self.Nobs
        ds_x_ref = x_obs_idx * scale_ref # Coordinates for Ref -> Obs
        ds_y_ref = y_obs_idx * scale_ref
        self.coords_ds_ref = cp.array([ds_y_ref, ds_x_ref])
        
        self._create_gnud_mask()

    def _create_gnud_mask(self):
        """Create spatial mask for observation grid nudging region"""
        scale_x = self.m.Nx / self.Nobs
        scale_y = self.m.Ny / self.Nobs
        
        # Expand observation grid to model grid with mask value 1
        self.gnud_mask = cp.zeros((self.m.Ny, self.m.Nx), dtype=np.float64) # Note (Ny, Nx)
        for i in range(self.Nobs):
            for j in range(self.Nobs):
                i_start = int(i * scale_x)
                i_end = int((i + 1) * scale_x)
                j_start = int(j * scale_y)
                j_end = int((j + 1) * scale_y)
                self.gnud_mask[j_start:j_end, i_start:i_end] = 1.0

    def _step_gnud(self):
        """Apply grid nudging
        
        Steps:
        1. Extract model and reference vorticity in physical space
        2. Downsample to observation grid
        3. Interpolate back to model grid
        4. Compute nudging forcing = mu * (reference - model)
        5. Apply spatial mask (1 on obs grid, 0 elsewhere)
        6. Convert to Fourier space and apply to model
        """
        q_m_r = ifft2(self.m.q_hat).real # back to physical space (real space)
        q_ref_r = ifft2(self.m_ref.q_hat).real

        q_m_sub = map_coordinates(q_m_r, self.coords_ds_m, order=0, mode='wrap')
        q_ref_sub = map_coordinates(q_ref_r, self.coords_ds_ref, order=0, mode='wrap')

        q_m_sub = q_m_sub - cp.mean(q_m_sub) # zero correction for periodic domain
        q_ref_sub = q_ref_sub - cp.mean(q_ref_sub)
         
        if self.interpolant == 'linear':
            self.Ih_m = self._linear_intp(q_m_sub)
            self.Ih_ref = self._linear_intp(q_ref_sub)
            
        elif self.interpolant == 'block':
            self.Ih_m = self._block_intp(q_m_sub)
            self.Ih_ref = self._block_intp(q_ref_sub)
            
        gnud_forcing = self.mu * (self.Ih_ref - self.Ih_m)
        gnud_forcing_masked = gnud_forcing * self.gnud_mask # apply mask: 1 on obs grid, 0 elsewhere
        self.m.cda_term = fft2(gnud_forcing_masked) # back to fourier space

    def _linear_intp(self, phi):
        """Linear interpolation from observation grid to model grid"""
        # using map_coordinate to deal with periodic boundary
        phi_intp = map_coordinates(phi, self.rcoord_model, order=1, mode='wrap', prefilter=False)
        return phi_intp.reshape(self.m.Nx, self.m.Ny)

    def _block_intp(self, phi):
        """Block/nearest-neighbor interpolation from observation grid to model grid"""
        # using map_coordinate to deal with periodic boundary
        phi_intp = map_coordinates(phi, self.rcoord_model, order=0, mode='wrap', prefilter=False)
        return phi_intp.reshape(self.m.Nx, self.m.Ny)

    def model_rmse(self, phi_m, phi_ref):
        """Compute RMSE between model and reference at observation grid"""
        step_ref = int(self.m_ref.Nx / self.m.Nx)
        phi_m_r = ifft2(phi_m).real # back to physical space (real space)
        phi_ref_r = ifft2(phi_ref).real
        phi_ref_sub = phi_ref_r[::step_ref, ::step_ref]
        if phi_ref_sub.shape != phi_m_r.shape:
            # Crop to the common overlap to prevent crash
            limit = min(phi_ref_sub.shape[0], phi_m_r.shape[0])
            phi_ref_sub = phi_ref_sub[:limit, :limit]
            phi_m_r = phi_m_r[:limit, :limit]
        diff = phi_m_r - phi_ref_sub
        m_rmse = cp.sqrt(cp.mean(diff**2))

        return m_rmse.get()

    def create_nc(self, nf):
        """Create netCDF output file with Grid Nudging diagnostics"""
        outdir = self.m.savedir
        # Create output filename with counter
        nc_filename = os.path.join(outdir, "output_%04d.nc"%(nf))

        if self.m.is_not_rst:
            self.m.create_nc(nf)
            self.gnudF_var = self.m.ds.createVariable('gnudF', 'f8', ('time', 'y', 'x'), zlib=False)
            self.ihm_var = self.m.ds.createVariable('Ihm', 'f8', ('time', 'y', 'x'), zlib=False)
            self.ihref_var = self.m.ds.createVariable('Ihref', 'f8', ('time', 'y', 'x'), zlib=False)
            self.qrmse_var = self.m.ds.createVariable('qrmse', 'f8', ('time',), zlib=False)
            self.tecdak_var = self.m.ds.createVariable('tecdak', 'f8', ('time', 'k'), zlib=False)
            self.tzcdak_var = self.m.ds.createVariable('tzcdak', 'f8', ('time', 'k'), zlib=False)
            self.fecdak_var = self.m.ds.createVariable('fecdak', 'f8', ('time', 'k'), zlib=False)
            self.fzcdak_var = self.m.ds.createVariable('fzcdak', 'f8', ('time', 'k'), zlib=False)
        else:
            self.m.create_nc(nf)
            self.gnudF_var = self.m.ds.variables['gnudF']
            self.ihm_var = self.m.ds.variables['Ihm']
            self.ihref_var = self.m.ds.variables['Ihref']
            self.qrmse_var = self.m.ds.variables['qrmse']
            self.tecdak_var = self.m.ds.variables['tecdak']
            self.tzcdak_var = self.m.ds.variables['tzcdak']
            self.fecdak_var = self.m.ds.variables['fecdak']
            self.fzcdak_var = self.m.ds.variables['fzcdak']

    def save_var(self, it):
        """Save variables to netCDF output"""
        self.m.save_var(it)
        # gnud term
        self.gnudF_var[it, :, :] = ifft2(self.m.cda_term).real.get()
        self.ihm_var[it, :, :] = self.Ih_m.get()
        self.ihref_var[it, :, :] = self.Ih_ref.get()
        self.qrmse_var[it] = self.model_rmse(self.m.q_hat, self.m_ref.q_hat)
        self.tecdak_var[it, :], self.tzcdak_var[it, :], self.fecdak_var[it, :], self.fzcdak_var[it, :] = self.m.get_diagCda(self.m.p_hat, self.m.q_hat, self.m.cda_term)
        self.m.ds.sync()

    def gnud_run(self, scheme='ab3', tmax=40, tsave=200, nsave=100, savedir='run_gnud', saveplot=False):
        """Main loop for grid nudging data assimilation
        
        Args:
            scheme: Time stepping scheme ('ab3' or 'rk4')
            tmax: Maximum simulation time
            tsave: Time steps between saves
            nsave: Number of saves per file
            savedir: Directory to save output
            saveplot: Whether to save diagnostic plots
        """
        self.m.ts_scheme = scheme
        self.m_ref.ts_scheme = scheme
        self.m.t = self.m.trst
        self.m_ref.t = self.m_ref.trst
        self.m.n_steps = 0
        self.m_ref.n_steps = 0

        self.m.savedir = savedir
        os.makedirs(self.m.savedir, exist_ok=True)

        total_steps = int(round(tmax / self.dt))

        print(f"Starting Grid Nudging. dTgnud={self.dTobs}, tmax={tmax}")
        print(f"Step interval -> Model: {self.intvl_model}, Ref: {self.intvl_ref}, Gnud: {self.intvl_gnud}")

        tsrst = int(1 / self.m.dt) # save rst every 1 time unit timestep
        nrst = nsave
        insave = nsave
        inrst = nrst
        nf = 0
        nfrst = 0

        for n in range(total_steps + 1):
            if n % self.intvl_gnud == 0:
                self._step_gnud()

            if n % self.intvl_model == 0:
                if self.m.n_steps % tsave == 0:

                    if insave == nsave:
                        itsave = 0 # time index -it
                        if nf > 0: self.m.ds.close()
                        self.create_nc(nf)
                        insave = 0 # save number index-in
                        nf += 1
                
                    self.save_var(itsave)
                    E_crt = self.m.get_Etot(self.m.p_hat) / self.m.Nx / self.m.Ny
                    Vrms_crt = self.m.get_Vrms(self.m.p_hat)
                    print(f"   step {self.m.n_steps:7d}  t={self.m.t:9.6f}s E={E_crt:.4e} Vrms={Vrms_crt:.4e}", end="\n")
                    if saveplot:
                        self.m.plot_diag()
                    itsave += 1
                    insave += 1
                if self.m.n_steps % tsrst == 0:
                    if inrst == nrst:
                        itrst = 0 # time index -it
                        if nfrst > 0: self.m.rstds.close()
                        self.m.create_rst(nfrst)
                        inrst = 0 # save number index-in
                        nfrst += 1

                    self.m.save_rst(itrst)
                    itrst += 1
                    inrst += 1


                self.m._step_forward()
                self.m.t += self.m.dt
                self.m.n_steps += 1

            if n % self.intvl_ref == 0:
                self.m_ref._step_forward()
                self.m_ref.t += self.m_ref.dt
                self.m_ref.n_steps += 1
        self.m.ds.close()
        
        print('Done.')
