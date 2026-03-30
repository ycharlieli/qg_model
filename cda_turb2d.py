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
    def __init__(self, m=None, m_ref=None, interpolant='block', Nobs=256, dTobs=0.1, mu=0.1,
                 obs_field='q', is_gnuding=True, is_not_rst=True, rtrst=0.0):
        """Initialize CDA with model and reference models"""
        self.m = m
        self.m_cda = copy.deepcopy(m)
        self.is_gnuding = is_gnuding
        if self.is_gnuding:
            self.m_gnud = copy.deepcopy(m)
        else:
            self.m_gnud = None
        self.m_ref = m_ref
        self.interpolant = interpolant
        self.Nobs = Nobs
        self.dTobs = dTobs
        self.obs_field = obs_field.lower()
        self.dt = np.min([self.m.dt, self.m_ref.dt])
        self.intvl_model = int(round(self.m.dt / self.dt))
        self.intvl_ref = int(round(self.m_ref.dt / self.dt))
        self.intvl_da = int(round(self.dTobs / self.dt))
        self.mu = mu
        self.errspec_eps = 1e-30
        self._init_grid()
        self.is_not_rst = is_not_rst
        self.rtrst = rtrst

    def _init_grid(self):
        """Initialize observation grid and coordinate mappings"""
        self.x_obs = cp.linspace(0, self.m.Lx, self.Nobs)
        self.y_obs = cp.linspace(0, self.m.Ly, self.Nobs)
        x_model_idx, y_model_idx = cp.meshgrid(cp.arange(self.m.Nx), cp.arange(self.m.Ny))
        ratio = self.Nobs / self.m.Nx
        rcoord_x = x_model_idx.ravel() * ratio
        rcoord_y = y_model_idx.ravel() * ratio
        self.rcoord_model = cp.array([rcoord_y, rcoord_x])
        x_obs_idx, y_obs_idx = cp.meshgrid(cp.arange(self.Nobs), cp.arange(self.Nobs))

        self.scale_m = self.m.Nx / self.Nobs
        ds_x_m = x_obs_idx * self.scale_m
        ds_y_m = y_obs_idx * self.scale_m
        self.coords_ds_m = cp.array([ds_y_m, ds_x_m])

        scale_ref = self.m_ref.Nx / self.Nobs
        ds_x_ref = x_obs_idx * scale_ref
        ds_y_ref = y_obs_idx * scale_ref
        self.coords_ds_ref = cp.array([ds_y_ref, ds_x_ref])

    def _get_obs_hat(self, model):
        """Return the spectral field used by the observation operator."""
        if self.obs_field == 'q':
            return model.q_hat
        return model.inversion * model.q_hat

    def _obs_to_q_hat(self, obs_field):
        """Map an observation-space field on the model grid into q-space."""
        obs_hat = fft2(obs_field)
        if self.obs_field == 'q':
            return obs_hat
        return (self.m_cda.lap - self.m_cda.gamma**2) * obs_hat

    def _donwsampling(self, phi_m_hat, phi_ref_hat):
        phi_m_r = ifft2(phi_m_hat).real
        phi_m_sub = map_coordinates(phi_m_r, self.coords_ds_m, order=0, mode='grid-wrap')
        phi_ref_r = ifft2(phi_ref_hat).real
        phi_ref_sub = map_coordinates(phi_ref_r, self.coords_ds_ref, order=0, mode='grid-wrap')
        phi_m_sub = phi_m_sub - cp.mean(phi_m_sub)
        phi_ref_sub = phi_ref_sub - cp.mean(phi_ref_sub)
        return phi_m_sub, phi_ref_sub

    def _step_cda(self):
        """Apply continuous data assimilation nudging."""
        phi_m_hat = self._get_obs_hat(self.m_cda)
        phi_ref_hat = self._get_obs_hat(self.m_ref)
        phi_m_sub, phi_ref_sub = self._donwsampling(phi_m_hat, phi_ref_hat)

        if self.interpolant == 'linear':
            self.Ih_m = self._linear_intp(phi_m_sub)
            self.Ih_ref = self._linear_intp(phi_ref_sub)
        else:
            self.Ih_m = self._block_intp(phi_m_sub)
            self.Ih_ref = self._block_intp(phi_ref_sub)

        self.Ih_m_q_hat = self._obs_to_q_hat(self.Ih_m)
        self.Ih_ref_q_hat = self._obs_to_q_hat(self.Ih_ref)
        self.cda_forcing = self.mu * (self.Ih_ref_q_hat - self.Ih_m_q_hat)
        self.m_cda.da_term = self.cda_forcing

    def _linear_intp(self, phi):
        """Linear interpolation from observation grid to model grid"""
        phi_intp = map_coordinates(phi, self.rcoord_model, order=1, mode='grid-wrap', prefilter=False)
        return phi_intp.reshape(self.m.Ny, self.m.Nx)

    def _block_intp(self, phi):
        """Block/nearest-neighbor interpolation from observation grid to model grid"""
        phi_intp = map_coordinates(phi, self.rcoord_model, order=0, mode='grid-wrap', prefilter=False)
        return phi_intp.reshape(self.m.Ny, self.m.Nx)

    def _step_gnud(self):
        """Apply strict grid nudging on observation points."""
        phi_m_hat = self._get_obs_hat(self.m_gnud)
        phi_ref_hat = self._get_obs_hat(self.m_ref)
        phi_m_sub, phi_ref_sub = self._donwsampling(phi_m_hat, phi_ref_hat)
        diff_obs = phi_ref_sub - phi_m_sub
        self.gnud_forcing = cp.zeros((self.m.Ny, self.m.Nx), dtype=np.float64)

        obs_i = cp.rint(cp.arange(self.Nobs) * self.scale_m).astype(cp.int64)
        obs_j = cp.rint(cp.arange(self.Nobs) * self.scale_m).astype(cp.int64)
        ii, jj = cp.meshgrid(obs_i, obs_j)

        self.gnud_forcing[jj, ii] = diff_obs
        self.Ih_gnud_q_hat = self._obs_to_q_hat(self.gnud_forcing)
        self.m_gnud.da_term = self.mu * self.Ih_gnud_q_hat

    def model_rmse(self, phi_m, phi_ref):
        """Compute RMSE between model and reference on the model grid."""
        step_ref = int(self.m_ref.Nx / self.m.Nx)
        phi_m_r = ifft2(phi_m).real
        phi_ref_r = ifft2(phi_ref).real
        phi_ref_sub = phi_ref_r[::step_ref, ::step_ref]
        if phi_ref_sub.shape != phi_m_r.shape:
            limit = min(phi_ref_sub.shape[0], phi_m_r.shape[0])
            phi_ref_sub = phi_ref_sub[:limit, :limit]
            phi_m_r = phi_m_r[:limit, :limit]
        diff = phi_m_r - phi_ref_sub
        m_rmse = cp.sqrt(cp.mean(diff**2))
        return m_rmse.get()

    def _to_mspace(self, phi_hat, source_model, target_model):
        """Return a spectral field represented on the target model grid."""
        if source_model.Nx == target_model.Nx and source_model.Ny == target_model.Ny:
            return phi_hat

        phi_r = ifft2(phi_hat).real
        step_y = int(source_model.Ny / target_model.Ny)
        step_x = int(source_model.Nx / target_model.Nx)
        phi_sub = phi_r[::step_y, ::step_x]
        if phi_sub.shape != (target_model.Ny, target_model.Nx):
            phi_sub = phi_sub[:target_model.Ny, :target_model.Nx]
        return fft2(phi_sub)

    def get_Eerrk(self, model):
        """Compute the normalized isotropic energy error spectrum."""
        p_ref_hat = self._to_mspace(self.m_ref.p_hat, self.m_ref, model)
        ek_model = np.asarray(model.get_Ek(model.p_hat))
        ek_ref = np.asarray(model.get_Ek(p_ref_hat))
        ek_err = np.asarray(model.get_Ek(model.p_hat - p_ref_hat))
        return ek_err / np.maximum(0.5 * (ek_model + ek_ref), self.errspec_eps)

    def get_Zerrk(self, model):
        """Compute the normalized isotropic enstrophy error spectrum."""
        q_ref_hat = self._to_mspace(self.m_ref.q_hat, self.m_ref, model)
        zk_model = np.asarray(model.get_Zk(model.q_hat))
        zk_ref = np.asarray(model.get_Zk(q_ref_hat))
        zk_err = np.asarray(model.get_Zk(model.q_hat - q_ref_hat))
        return zk_err / np.maximum(0.5 * (zk_model + zk_ref), self.errspec_eps)

    def create_ctrl_nc(self, nf):
        self.m.create_nc(nf, prefix='ctrl_o')
        if 'rmse' not in self.m.ds.variables:
            self.rmse_ctrl_var = self.m.ds.createVariable('rmse', 'f8', ('time',))
        else:
            self.rmse_ctrl_var = self.m.ds.variables['rmse']
        if 'eerrk' not in self.m.ds.variables:
            self.eerrk_ctrl_var = self.m.ds.createVariable('eerrk', 'f8', ('time', 'k'))
        else:
            self.eerrk_ctrl_var = self.m.ds.variables['eerrk']
        if 'zerrk' not in self.m.ds.variables:
            self.zerrk_ctrl_var = self.m.ds.createVariable('zerrk', 'f8', ('time', 'k'))
        else:
            self.zerrk_ctrl_var = self.m.ds.variables['zerrk']

    def create_cda_nc(self, nf):
        self.m_cda.create_nc(nf, prefix='cda_o')
        if 'rmse' not in self.m_cda.ds.variables:
            self.rmse_cda_var = self.m_cda.ds.createVariable('rmse', 'f8', ('time',))
        else:
            self.rmse_cda_var = self.m_cda.ds.variables['rmse']
        if 'Ihm' not in self.m_cda.ds.variables:
            self.Ih_m_var = self.m_cda.ds.createVariable('Ihm', 'f8', ('time', 'y', 'x'))
        else:
            self.Ih_m_var = self.m_cda.ds.variables['Ihm']
        if 'Ihref' not in self.m_cda.ds.variables:
            self.Ih_ref_var = self.m_cda.ds.createVariable('Ihref', 'f8', ('time', 'y', 'x'))
        else:
            self.Ih_ref_var = self.m_cda.ds.variables['Ihref']
        if 'eerrk' not in self.m_cda.ds.variables:
            self.eerrk_cda_var = self.m_cda.ds.createVariable('eerrk', 'f8', ('time', 'k'))
        else:
            self.eerrk_cda_var = self.m_cda.ds.variables['eerrk']
        if 'zerrk' not in self.m_cda.ds.variables:
            self.zerrk_cda_var = self.m_cda.ds.createVariable('zerrk', 'f8', ('time', 'k'))
        else:
            self.zerrk_cda_var = self.m_cda.ds.variables['zerrk']

    def create_gnud_nc(self, nf):
        if self.is_gnuding:
            self.m_gnud.create_nc(nf,prefix='gnud_o')
            if 'rmse' not in self.m_gnud.ds.variables:
                self.rmse_gnud_var = self.m_gnud.ds.createVariable('rmse', 'f8', ('time',))
            else:
                self.rmse_gnud_var = self.m_gnud.ds.variables['rmse']
            if 'eerrk' not in self.m_gnud.ds.variables:
                self.eerrk_gnud_var = self.m_gnud.ds.createVariable('eerrk', 'f8', ('time', 'k'))
            else:
                self.eerrk_gnud_var = self.m_gnud.ds.variables['eerrk']
            if 'zerrk' not in self.m_gnud.ds.variables:
                self.zerrk_gnud_var = self.m_gnud.ds.createVariable('zerrk', 'f8', ('time', 'k'))
            else:
                self.zerrk_gnud_var = self.m_gnud.ds.variables['zerrk']

    def create_ref_nc(self, nf):
        self.m_ref.create_nc(nf, prefix='ref_o')

    def create_nc(self,nf):
        self.create_ctrl_nc(nf)
        self.create_cda_nc(nf)
        self.create_gnud_nc(nf)
        self.create_ref_nc(nf)

    def create_ctrl_rst(self, nf):
        self.m.create_rst(nf, prefix='ctrl_r') 
        self._bind_tcda_var()

    def _bind_tcda_var(self):
        if 'tcda' not in self.m.rstds.variables:
            self.tcda_var = self.m.rstds.createVariable('tcda', 'f8', ('time',))
        else:
            self.tcda_var = self.m.rstds.variables['tcda']

    def create_cda_rst(self, nf):     
        self.m_cda.create_rst(nf, prefix='cda_r')
        
    def create_gnud_rst(self, nf):
        if self.is_gnuding:
            self.m_gnud.create_rst(nf, prefix='gnud_r')
            
    def create_ref_rst(self, nf):
        self.m_ref.create_rst(nf, prefix='ref_r')
        
    def create_rst(self, nf):
        self.create_ctrl_rst(nf)
        self.m_cda.create_rst(nf, prefix='cda_r')
        if self.is_gnuding:
            self.m_gnud.create_rst(nf, prefix='gnud_r')
        self.m_ref.create_rst(nf, prefix='ref_r')

    def close_nc(self):
        self.m.ds.close()
        self.m_cda.ds.close()
        if self.is_gnuding:
            self.m_gnud.ds.close()
        self.m_ref.ds.close()

    def close_rst(self):
        self.m.rstds.close()
        self.m_cda.rstds.close()
        if self.is_gnuding:
            self.m_gnud.rstds.close()
        self.m_ref.rstds.close()

    def save_var(self, it):
        """Save variables to netCDF output"""
        self.m.save_var(it)
        self.rmse_ctrl_var[it] = self.model_rmse(self.m.q_hat, self.m_ref.q_hat)
        self.eerrk_ctrl_var[it, :] = self.get_Eerrk(self.m)
        self.zerrk_ctrl_var[it, :] = self.get_Zerrk(self.m)
        self.m.ds.sync()

        self.m_cda.save_var(it)
        self.rmse_cda_var[it] = self.model_rmse(self.m_cda.q_hat, self.m_ref.q_hat)
        self.Ih_m_var[it, :, :] = self.Ih_m.get()
        self.Ih_ref_var[it, :, :] = self.Ih_ref.get()
        self.eerrk_cda_var[it, :] = self.get_Eerrk(self.m_cda)
        self.zerrk_cda_var[it, :] = self.get_Zerrk(self.m_cda)
        self.m_cda.ds.sync()

        if self.is_gnuding:
            self.m_gnud.save_var(it)
            self.rmse_gnud_var[it] = self.model_rmse(self.m_gnud.q_hat, self.m_ref.q_hat)
            self.eerrk_gnud_var[it, :] = self.get_Eerrk(self.m_gnud)
            self.zerrk_gnud_var[it, :] = self.get_Zerrk(self.m_gnud)
            self.m_gnud.ds.sync()

        self.m_ref.save_var(it)
        self.m_ref.ds.sync()
    def save_rst(self,it):
        """Save restart files"""
        self.m.save_rst(it)
        self.tcda_var[it] = self.rt
        self.m.rstds.sync()
        self.m_cda.save_rst(it)
        if self.is_gnuding:
            self.m_gnud.save_rst(it)
        self.m_ref.save_rst(it)

    def _set_model_times(self, rt):
        """Set absolute model time = pickup time from spinup + CDA relative time."""
        self.m.t = self.m.trst + rt
        self.m_cda.t = self.m_cda.trst + rt
        if self.is_gnuding:
            self.m_gnud.t = self.m_gnud.trst + rt
        self.m_ref.t = self.m_ref.trst + rt



    def cda_run(self,scheme='ab3',tmax=40,tsave=200,tsave_rst=2000,nsave=100,savedir='run_cda0',saveplot=False):
        """Main loop for continuous data assimilation
        
        Args:
            scheme: Time stepping scheme ('ab3' or 'rk4')
            tmax: Maximum simulation time
            tsave: Time steps between saves
            tsave_rst: Steps between restart checkpoint saves
            nsave: Number of saves per file
            savedir: Directory to save output
            saveplot: Whether to save diagnostic plots
        """
        self.m.ts_scheme     = scheme
        self.m_cda.ts_scheme = scheme
        if self.is_gnuding:
            self.m_gnud.ts_scheme = scheme
        self.m_ref.ts_scheme = scheme

        self.m.savedir = savedir
        self.m_cda.savedir = savedir
        if self.is_gnuding:
            self.m_gnud.savedir = savedir
        self.m_ref.savedir = savedir
        os.makedirs(self.m.savedir, exist_ok=True)
        self.tsave_rst = tsave_rst

        total_steps = int(round(tmax/self.dt))

        print(f"Starting CDA. dTobs={self.dTobs}, tmax={tmax}")
        print(f"Step interval -> Model: {self.intvl_model}, Ref: {self.intvl_ref}, Obs: {self.intvl_da}")

        nrst = nsave
        # Initialize or continue from restart time
        if self.is_not_rst:
            nf0 = 0
            nfrst0 = 0
            itsave = 0
            itrst = 0
            insave = nsave
            inrst = nrst
            n_start = 0
            nf = nf0
            nfrst = nfrst0            
        else:
            # Calculate which file and position to resume from
            n_start_idx = int(round(self.rtrst / self.dt))
            hist_saves = n_start_idx // tsave
            nf0 = hist_saves // nsave
            itsave = hist_saves % nsave
            hist_saves_rst = n_start_idx // tsave_rst
            nfrst0 = hist_saves_rst // nrst
            itrst = hist_saves_rst % nrst
            nf = nf0
            nfrst = nfrst0
            # Setup initial file state for restart
            if itsave == 0:
                insave = nsave  # Force create_nc next step
            else:
                insave = itsave
                # Open existing file for appending since we're mid-file
                self.create_nc(nf0)
                nf+=1
            # Setup initial restart file state 
            if itrst == 0:
                inrst = nrst # Force create_rst next step
            else:
                inrst = itrst
                # Open existing file for appending since we're mid-file
                self.create_rst(nfrst0)
                nfrst+=1
                
            n_start = n_start_idx

        # Project runtime to the discrete global step grid to avoid drift.
        self.rt = n_start * self.dt
        # Model time is absolute and continues from spinup pickup time.
        # rt is CDA-relative runtime used only for CDA restart bookkeeping.
        self._set_model_times(self.rt)

        for n in range(n_start,total_steps+1):
            self.rt = n * self.dt
            self.m.n_steps = n
            self.m_cda.n_steps = n
            if self.is_gnuding:
                self.m_gnud.n_steps = n
            self.m_ref.n_steps = n
            if n % self.intvl_da == 0:
                self._step_cda()
                if self.is_gnuding:
                    self._step_gnud()
            # print diagnostics to console every 10,000 steps
            if n%10000 == 0:
                # Compute and print energy and diagnostic statistics
                E_ctrl = self.m.get_Etot(self.m.p_hat)/self.m.Nx/self.m.Ny
                E_cda = self.m_cda.get_Etot(self.m_cda.p_hat)/self.m_cda.Nx/self.m_cda.Ny
                E_gnud = 0.0
                if self.is_gnuding:
                    E_gnud = self.m_gnud.get_Etot(self.m_gnud.p_hat)/self.m_gnud.Nx/self.m_gnud.Ny
                E_ref = self.m_ref.get_Etot(self.m_ref.p_hat)/self.m_ref.Nx/self.m_ref.Ny
                import time
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}      step {self.m.n_steps:7d}  t={self.m.t:9.6f}s E_ctrl={E_ctrl:.4e} E_cda={E_cda:.4e} E_gnud={E_gnud:.4e} E_ref={E_ref:.4e}", end="\n")
            if n % self.intvl_model ==0:
                if self.m.n_steps % tsave == 0:

                    if insave == nsave:
                        itsave =0 #time index -it
                        if nf > nf0 : 
                            self.close_nc()
                        self.create_nc(nf)
                        insave=0 #save number index-in
                        nf+=1
                
                    self.save_var(itsave)
                    import time
                    print(f"[save_var] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.m.n_steps:7d}  t={self.m.t:9.6f}s ", end="\n")
                
                    
                    if saveplot:
                        self.m.plot_diag()
                    itsave +=1
                    insave +=1  
                if self.m.n_steps % tsave_rst==0:
                    if inrst == nrst:
                        itrst =0 #time index -it
                        if nfrst > nfrst0 : 
                            self.close_rst()
                        self.create_rst(nfrst)
                        inrst=0 #save number index-in
                        nfrst+=1

                    self.save_rst(itrst)
                    import time
                    print(f"[save_rst] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.m.n_steps:7d}  t={self.m.t:9.6f}s ", end="\n")
                
                    itrst +=1
                    inrst +=1


                self.m._step_forward()
                self.m_cda._step_forward() 
                if self.is_gnuding:
                    self.m_gnud._step_forward()
                # Reconstruct absolute model clocks from integer step index.
                # This avoids cumulative floating-point drift from repeated += dt.
                self.m.t = self.m.trst + (n + self.intvl_model) * self.dt
                self.m_cda.t = self.m_cda.trst + (n + self.intvl_model) * self.dt
                if self.is_gnuding:
                    self.m_gnud.t = self.m_gnud.trst + (n + self.intvl_model) * self.dt

            if n % self.intvl_ref ==0:
                self.m_ref._step_forward()
                self.m_ref.t = self.m_ref.trst + (n + self.intvl_ref) * self.dt
        
        self.close_nc()
        self.close_rst()

        print('Done.')
