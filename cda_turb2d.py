import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.ndimage import map_coordinates
from scipy.fft import fft2,ifft2,fftshift
from scipy.fft import set_global_backend
set_global_backend(cufft)
import os
import copy

class QGCDA:
    """Continuous Data Assimilation: Nudge model towards reference on full domain"""
    def __init__(self,m=None,m_ref=None,interpolant='linear',Nobs=256,dTobs=0.1,mu=0.1,is_gnuding=True):
        """Initialize CDA with model and reference models
        
        Args:
            m: Model instance (used as template and operator engine)
            m_ref: Reference/truth model
            interpolant: 'linear' or 'block' interpolation method
            Nobs: Number of observation grid points (Nobs x Nobs)
            dTobs: Time interval between observations
            mu: Nudging strength parameter
            is_gnuding: Boolean to enable/disable Grid Nudging (GNUD)
        """
        self.m = m
        self.m_cda = copy.deepcopy(m)
        self.is_gnuding = is_gnuding
        if self.is_gnuding:
            self.m_gnud = copy.deepcopy(m)
        else:
            self.m_gnud = None
        self.m_ref = m_ref
        self.interpolant =  interpolant
        self.Nobs = Nobs
        self.dTobs = dTobs
        self.dt = np.min([self.m.dt,self.m_ref.dt])
        self.intvl_model = int(round(self.m.dt/self.dt)) # check step instead time to avoid trunction error
        self.intvl_ref = int(round(self.m_ref.dt/self.dt))
        self.intvl_da = int(round(self.dTobs/self.dt))
        self.mu = mu
        self._init_grid()

    def _init_grid(self):
        """Initialize observation grid and coordinate mappings"""
        self.x_obs = cp.linspace(0,self.m.Lx,self.Nobs) # position of observation
        self.y_obs = cp.linspace(0,self.m.Ly,self.Nobs)
        x_model_idx, y_model_idx = cp.meshgrid(cp.arange(self.m.Nx),cp.arange(self.m.Ny))
        ratio = self.Nobs/self.m.Nx
        rcoord_x =  x_model_idx.ravel() * ratio# relative coordinate of model to observation
        rcoord_y =  y_model_idx.ravel() * ratio
        self.rcoord_model = cp.array([rcoord_y,rcoord_x])
        x_obs_idx, y_obs_idx = cp.meshgrid(cp.arange(self.Nobs), cp.arange(self.Nobs))
        
        self.scale_m = self.m.Nx / self.Nobs
        ds_x_m = x_obs_idx * self.scale_m# Coordinates for Model -> Obs
        ds_y_m = y_obs_idx * self.scale_m
        self.coords_ds_m = cp.array([ds_y_m, ds_x_m])
        
        
        scale_ref = self.m_ref.Nx / self.Nobs
        ds_x_ref = x_obs_idx * scale_ref# Coordinates for Ref -> Obs
        ds_y_ref = y_obs_idx * scale_ref
        self.coords_ds_ref = cp.array([ds_y_ref, ds_x_ref])

    def _donwsampling(self,phi):
        phi_m_r = ifft2(phi).real # back to physical space (real space)
        phi_m_sub = map_coordinates(phi_m_r,self.coords_ds_m,order=0, mode='wrap')
        phi_ref_r = ifft2(self.m_ref.q_hat).real
        phi_ref_sub = map_coordinates(phi_ref_r,self.coords_ds_ref,order=0,mode='wrap')
        phi_m_sub = phi_m_sub - cp.mean(phi_m_sub) #zero correction for periodic domain
        phi_ref_sub = phi_ref_sub - cp.mean(phi_ref_sub)
        return phi_m_sub, phi_ref_sub

    def _step_cda(self):
        """Apply continuous data assimilation nudging
        
        Steps:
        1. Extract model and reference vorticity in physical space
        2. Downsample to observation grid
        3. Interpolate back to model grid
        4. Compute nudging forcing = mu * (reference - model)
        5. Convert to Fourier space and apply to model
        """
        q_m_sub, q_ref_sub = self._donwsampling(self.m_cda.q_hat)
         
        if self.interpolant == 'linear':
            self.Ih_m = self._linear_intp(q_m_sub)
            self.Ih_ref = self._linear_intp(q_ref_sub)
            
        elif self.interpolant == 'block':
            self.Ih_m = self._block_intp(q_m_sub)
            self.Ih_ref = self._block_intp(q_ref_sub)
            
        self.cda_forcing = self.mu*(self.Ih_ref-self.Ih_m)
        self.m_cda.da_term = fft2(self.cda_forcing)# back to fourier space

    def _linear_intp(self,phi):
        """Linear interpolation from observation grid to model grid"""
        # using map_coordinate to deal with periodic  boundary
        phi_intp = map_coordinates(phi,self.rcoord_model,order=1,mode='wrap',prefilter=False)
        return phi_intp.reshape(self.m.Ny,self.m.Nx)

    def _block_intp(self,phi):
        """Block/nearest-neighbor interpolation from observation grid to model grid"""
        # using map_coordinate to deal with periodic  boundary
        phi_intp = map_coordinates(phi,self.rcoord_model,order=0,mode='wrap',prefilter=False)
        return phi_intp.reshape(self.m.Ny,self.m.Nx)
    
    def _step_gnud(self):
        """Apply grid nudging (not implemented)"""
        # only apply nudging at observation grid position at model grid
        q_m_sub, q_ref_sub = self._donwsampling(self.m_gnud.q_hat)
        # Strict grid nudging: non-zero only at observation points
        diff_obs = q_ref_sub - q_m_sub
        self.gnud_forcing = cp.zeros((self.m.Ny, self.m.Nx), dtype=np.float64)  # Note (Ny, Nx)

        # Scale observation indices to model grid indices (Nearest Neighbor)
        obs_i = cp.rint(cp.arange(self.Nobs) * self.scale_m).astype(cp.int64)
        obs_j = cp.rint(cp.arange(self.Nobs) * self.scale_m).astype(cp.int64)
        ii, jj = cp.meshgrid(obs_i, obs_j)

        self.gnud_forcing[jj, ii] = self.mu * diff_obs
        self.m_gnud.da_term = fft2(self.gnud_forcing)# back to fourier space  

    def model_rmse(self,phi_m,phi_ref):
        """Compute RMSE between model and reference at observation grid"""
        step_ref = int(self.m_ref.Nx/self.m.Nx)
        phi_m_r = ifft2(phi_m).real # back to physical space (real space)
        phi_ref_r = ifft2(phi_ref).real
        phi_ref_sub = phi_ref_r[::step_ref,::step_ref]
        if phi_ref_sub.shape != phi_m_r.shape:
            # Crop to the common overlap to prevent crash
            limit = min(phi_ref_sub.shape[0], phi_m_r.shape[0])
            phi_ref_sub = phi_ref_sub[:limit, :limit]
            phi_m_r = phi_m_r[:limit, :limit]
        diff = phi_m_r - phi_ref_sub
        m_rmse = cp.sqrt(cp.mean(diff**2))

        return m_rmse.get()


    def create_dns_nc(self,nf):
        self.m.create_nc(nf,prefix='dns')
        if 'rmse' not in self.m.ds.variables:
            self.rmse_dns_var = self.m.ds.createVariable('rmse', 'f8', ('time',))
        else:
            self.rmse_dns_var = self.m.ds.variables['rmse']


    def create_cda_nc(self,nf):
        self.m_cda.create_nc(nf,prefix='cda')
        if 'rmse' not in self.m_cda.ds.variables:
            self.rmse_cda_var = self.m_cda.ds.createVariable('rmse', 'f8', ('time',))
            self.Ih_m_var = self.m_cda.ds.createVariable('Ihm', 'f8', ('time', 'y', 'x'))
            self.Ih_ref_var = self.m_cda.ds.createVariable('Ihref', 'f8', ('time', 'y', 'x'))
        else:
            self.rmse_cda_var = self.m_cda.ds.variables['rmse']
            self.Ih_m_var = self.m_cda.ds.variables['Ihm']
            self.Ih_ref_var = self.m_cda.ds.variables['Ihref']

    def create_gnud_nc(self,nf):
        if self.is_gnuding:
            self.m_gnud.create_nc(nf,prefix='gnud')
            if 'rmse' not in self.m_gnud.ds.variables:
                self.rmse_gnud_var = self.m_gnud.ds.createVariable('rmse', 'f8', ('time',))
            else:
                self.rmse_gnud_var = self.m_gnud.ds.variables['rmse']
   
    def create_ref_nc(self,nf):
        self.m_ref.create_nc(nf,prefix='ref')

    def create_nc(self,nf):
        self.create_dns_nc(nf)
        self.create_cda_nc(nf)
        self.create_gnud_nc(nf)
        self.create_ref_nc(nf)

    def close_nc(self):
        self.m.ds.close()
        self.m_cda.ds.close()
        if self.is_gnuding:
            self.m_gnud.ds.close()
        self.m_ref.ds.close()

    def save_var(self,it):
        """Save variables to netCDF output"""
        self.m.save_var(it)
        self.rmse_dns_var[it] = self.model_rmse(self.m.q_hat, self.m_ref.q_hat)
        self.m.ds.sync()

        self.m_cda.save_var(it)
        self.rmse_cda_var[it] = self.model_rmse(self.m_cda.q_hat, self.m_ref.q_hat)
        self.Ih_m_var[it,:,:] = self.Ih_m.get()
        self.Ih_ref_var[it,:,:] = self.Ih_ref.get()
        self.m_cda.ds.sync()

        if self.is_gnuding:
            self.m_gnud.save_var(it)
            self.rmse_gnud_var[it] = self.model_rmse(self.m_gnud.q_hat, self.m_ref.q_hat)
            self.m_gnud.ds.sync()

        self.m_ref.save_var(it)
    # def save_ref(self,ds):



    def cda_run(self,scheme='ab3',tmax=40,tsave=200,nsave=100,savedir='run_cda0',saveplot=False):
        """Main loop for continuous data assimilation
        
        Args:
            scheme: Time stepping scheme ('ab3' or 'rk4')
            tmax: Maximum simulation time
            tsave: Time steps between saves
            nsave: Number of saves per file
            savedir: Directory to save output
            saveplot: Whether to save diagnostic plots
        """
        self.m.ts_scheme     = scheme
        self.m_cda.ts_scheme = scheme
        if self.is_gnuding:
            self.m_gnud.ts_scheme = scheme
        self.m_ref.ts_scheme = scheme
        
        self.m.t = self.m.trst
        self.m_cda.t = self.m.trst # Start from same time as m
        if self.is_gnuding:
            self.m_gnud.t = self.m.trst
        self.m_ref.t = self.m_ref.trst 
        
        self.m.n_steps = 0
        self.m_cda.n_steps = 0
        if self.is_gnuding:
            self.m_gnud.n_steps = 0
        self.m_ref.n_steps = 0

        self.m.savedir = savedir
        self.m_cda.savedir = savedir
        if self.is_gnuding:
            self.m_gnud.savedir = savedir
        self.m_ref.savedir = savedir
        os.makedirs(self.m.savedir, exist_ok=True)

        total_steps = int(round(tmax/self.dt))

        print(f"Starting CDA. dTobs={self.dTobs}, tmax={tmax}")
        print(f"Step interval -> Model: {self.intvl_model}, Ref: {self.intvl_ref}, Obs: {self.intvl_da}")

        # tsrst = int(1/self.m.dt) # save rst every 1  time unit timestep
        # nrst = nsave
        insave=nsave
        # inrst = nrst
        nf=0
        # nfrst=0

        for n in range(total_steps+1):
            if n % self.intvl_da == 0:
                self._step_cda()
                if self.is_gnuding:
                    self._step_gnud()

            if n % self.intvl_model ==0:
                if self.m.n_steps % tsave == 0:

                    if insave == nsave:
                        itsave =0 #time index -it
                        if nf > 0 : 
                            self.close_nc()
                        
                        self.create_nc(nf)
                        
                        insave=0 #save number index-in
                        nf+=1
                
                    self.save_var(itsave)
                    
                    E_dns = self.m.get_Etot(self.m.p_hat)/self.m.Nx/self.m.Ny
                    E_cda = self.m_cda.get_Etot(self.m_cda.p_hat)/self.m_cda.Nx/self.m_cda.Ny
                    E_gnud = 0.0
                    if self.is_gnuding:
                        E_gnud = self.m_gnud.get_Etot(self.m_gnud.p_hat)/self.m_gnud.Nx/self.m_gnud.Ny
                    E_ref = self.m_ref.get_Etot(self.m_ref.p_hat)/self.m_ref.Nx/self.m_ref.Ny

                    Vrms_crt = self.m.get_Vrms(self.m.p_hat)
                    print(f"   step {self.m.n_steps:7d}  t={self.m.t:9.6f}s E_dns={E_dns:.4e} E_cda={E_cda:.4e} E_gnud={E_gnud:.4e} E_ref={E_ref:.4e} Vrms={Vrms_crt:.4e}", end="\n")
                    if saveplot:
                        self.m.plot_diag()
                    itsave +=1
                    insave +=1  
                # if self.m.n_steps % tsrst==0:
                #     if inrst == nrst:
                #         itrst =0 #time index -it
                        
                #         if nfrst > 0 : 
                #             self.m.rstds.close()
                #             self.m_cda.rstds.close()
                #             self.m_gnud.rstds.close()
                #             self.m_ref.rstds.close()
                            
                #         self.m.create_rst(nfrst)
                #         self.m_cda.create_rst(nfrst)
                #         self.m_gnud.create_rst(nfrst)
                #         self.m_ref.create_rst(nfrst)
                        
                #         inrst=0 #save number index-in
                #         nfrst+=1

                #     self.m.save_rst(itrst)
                #     self.m_cda.save_rst(itrst)
                #     self.m_gnud.save_rst(itrst)
                #     self.m_ref.save_rst(itrst)
                    
                #     itrst +=1
                #     inrst +=1


                self.m._step_forward()
                self.m_cda._step_forward() 
                if self.is_gnuding:
                    self.m_gnud._step_forward()
                self.m.n_steps += 1
                self.m_cda.n_steps += 1
                if self.is_gnuding:
                    self.m_gnud.n_steps += 1
                self.m.t = self.m.trst + self.m.n_steps * self.m.dt
                self.m_cda.t = self.m_cda.trst + self.m_cda.n_steps * self.m_cda.dt
                if self.is_gnuding:
                    self.m_gnud.t = self.m_gnud.trst + self.m_gnud.n_steps * self.m_gnud.dt

            if n % self.intvl_ref ==0:
                self.m_ref._step_forward()
                self.m_ref.n_steps +=1
                self.m_ref.t = self.m_ref.trst + self.m_ref.n_steps * self.m_ref.dt
        
        self.close_nc()
        
        # try:
        #     self.m.rstds.close()
        #     self.m_cda.rstds.close()
        #     self.m_gnud.rstds.close()
        #     self.m_ref.rstds.close()
        # except:
        #     pass
        
        print('Done.')
