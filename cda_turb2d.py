import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.ndimage import map_coordinates
from scipy.fft import fft2,ifft2,fftshift
from scipy.fft import set_global_backend
set_global_backend(cufft)
import os
import copy
import time
import numpy_groupies as npg

class QGCDA:
    """Continuous Data Assimilation: Nudge model towards reference on full domain"""
    def __init__(self, m=None, m_ref=None, interpolant='block', Nobs=256, dTobs=0.1, mu=0.1,
                 obs_field='q', is_gnuding=False, is_not_rst=True, rtrst=0.0,
                 obs_spectral_filter=False, is_ctrl=False):
        """Initialize CDA with model and reference models

        Args:
            interpolant: Observation operator. Use 'linear'/'block' for
                subsample-then-interpolate, or 'spec' for direct spectral
                low-pass assimilation with k_cut = Nobs. Use 'ot2003' for
                exact low-mode insertion with the same spectral cutoff. Use
                'zeropad' for coarse-grid sampled observations reconstructed
                by spectral zero-padding up to the observation Nyquist.
            is_gnuding: If True, also run and save the strict grid-nudging
                comparison trajectory.
            obs_spectral_filter: If True, low-pass filter mapped observations on the
                model grid so only scales larger than the forcing scale are retained.
            is_ctrl: If True, also run and save the unnudged control trajectory.
        """
        if m is not None and m_ref is not None and m.precision != m_ref.precision:
            raise ValueError(
                f"Model and reference precisions differ: {m.precision!r} vs "
                f"{m_ref.precision!r}. Build both with the same precision.")
        self.m = m
        self.m_cda = copy.deepcopy(m)
        self.is_ctrl = is_ctrl
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
        self.obs_spectral_filter = obs_spectral_filter
        self.errspec_eps = 1e-30
        self.Ih_m = cp.zeros((self.m.Ny, self.m.Nx), dtype=cp.float64)
        self.Ih_ref = cp.zeros_like(self.Ih_m)
        self._shell_cache = {}
        self._init_grid()
        self.is_not_rst = is_not_rst
        self.rtrst = rtrst

    def _init_grid(self):
        """Initialize observation grid and coordinate mappings"""
        self.x_obs = cp.linspace(0, self.m.Lx, self.Nobs, endpoint=False)
        self.y_obs = cp.linspace(0, self.m.Ly, self.Nobs, endpoint=False)
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

        if self.obs_spectral_filter:
            self.obs_filter_mask = (self.m.kk <= float(self.m.fscale)).astype(self.m.rdtype)
        else:
            self.obs_filter_mask = None

        self.spec_k_cut = float(self.Nobs)
        self.spec_filter_mask_m = self._build_spec_filter_mask(self.m_cda)
        self.spec_filter_mask_ref = self._build_spec_filter_mask(self.m_ref)
        self.spec_filter_bool_m = self.spec_filter_mask_m.astype(cp.bool_)

    def _build_spec_filter_mask(self, model):
        """Return a low-pass mask using |k| < Nobs."""
        k_radius = cp.sqrt(model.nx2d.astype(cp.float64)**2
                           + model.ny2d.astype(cp.float64)**2)
        return (k_radius < self.spec_k_cut).astype(model.rdtype)

    def _get_obs_hat(self, model):
        """Return the spectral field used by the observation operator."""
        if self.obs_field == 'q':
            return model.q_hat
        return model.inversion * model.q_hat

    def _obs_hat_to_q_hat(self, obs_hat):
        """Convert a spectral observation field on the model grid into q-space."""
        if self.obs_field == 'q':
            return obs_hat
        return (self.m_cda.lap - self.m_cda.gamma**2) * obs_hat

    def _obs_to_q_hat(self, obs_field):
        """Map an observation-space field on the model grid into q-space."""
        obs_hat = fft2(obs_field)
        return self._obs_hat_to_q_hat(obs_hat)

    def _donwsampling(self, phi_m_hat, phi_ref_hat):
        phi_m_r = ifft2(phi_m_hat).real
        phi_m_sub = map_coordinates(phi_m_r, self.coords_ds_m, order=0, mode='grid-wrap')
        phi_ref_r = ifft2(phi_ref_hat).real
        phi_ref_sub = map_coordinates(phi_ref_r, self.coords_ds_ref, order=0, mode='grid-wrap')
        phi_m_sub = phi_m_sub - cp.mean(phi_m_sub)
        phi_ref_sub = phi_ref_sub - cp.mean(phi_ref_sub)
        return phi_m_sub, phi_ref_sub

    def _project_hat_to_model_grid(self, phi_hat, source_model, target_model):
        """Crop/pad low-wavenumber Fourier modes onto the target model grid."""
        if source_model.Nx == target_model.Nx and source_model.Ny == target_model.Ny:
            return phi_hat

        phi_hat_shift = cp.fft.fftshift(phi_hat)
        phi_hat_target_shift = cp.zeros((target_model.Ny, target_model.Nx), dtype=phi_hat.dtype)

        copy_ny = min(source_model.Ny, target_model.Ny)
        copy_nx = min(source_model.Nx, target_model.Nx)
        src_y0 = (source_model.Ny - copy_ny) // 2
        src_x0 = (source_model.Nx - copy_nx) // 2
        tgt_y0 = (target_model.Ny - copy_ny) // 2
        tgt_x0 = (target_model.Nx - copy_nx) // 2

        phi_hat_target_shift[tgt_y0:tgt_y0 + copy_ny, tgt_x0:tgt_x0 + copy_nx] = (
            phi_hat_shift[src_y0:src_y0 + copy_ny, src_x0:src_x0 + copy_nx]
        )

        scale = (target_model.Nx * target_model.Ny) / (source_model.Nx * source_model.Ny)
        return cp.fft.ifftshift(phi_hat_target_shift) * scale

    def _zeropad_intp(self, phi_obs):
        """Lift a coarse observation field to the model grid by spectral zero-padding."""
        phi_obs_hat = fft2(phi_obs)
        phi_obs_hat_shift = cp.fft.fftshift(phi_obs_hat)
        phi_hat_model_shift = cp.zeros((self.m_cda.Ny, self.m_cda.Nx), dtype=phi_obs_hat.dtype)

        src_ny, src_nx = phi_obs.shape
        tgt_ny, tgt_nx = self.m_cda.Ny, self.m_cda.Nx
        src_y0 = (tgt_ny - src_ny) // 2
        src_x0 = (tgt_nx - src_nx) // 2
        phi_hat_model_shift[src_y0:src_y0 + src_ny, src_x0:src_x0 + src_nx] = phi_obs_hat_shift

        scale = (tgt_nx * tgt_ny) / (src_nx * src_ny)
        phi_hat_model = cp.fft.ifftshift(phi_hat_model_shift) * scale
        phi_model = ifft2(phi_hat_model).real
        return phi_model - cp.mean(phi_model), phi_hat_model

    def _spec_intp(self, phi_hat, source_model, spec_filter_mask):
        """Apply the direct spectral observation operator with k_cut = Nobs."""
        phi_hat_lp = phi_hat * spec_filter_mask
        phi_hat_lp = self._project_hat_to_model_grid(phi_hat_lp, source_model, self.m_cda)
        phi_lp = ifft2(phi_hat_lp).real
        return phi_lp - cp.mean(phi_lp), phi_hat_lp

    def _ot2003_insert(self, phi_m_hat, phi_ref_hat):
        """Replace observed low modes exactly following the OT2003-style split."""
        spec_filter_mask = self.spec_filter_mask_m
        phi_ref_hat_model = self._project_hat_to_model_grid(phi_ref_hat, self.m_ref, self.m_cda)

        phi_m_obs_hat = phi_m_hat * spec_filter_mask
        phi_ref_obs_hat = phi_ref_hat_model * spec_filter_mask
        phi_inserted_hat = phi_m_hat.copy()
        cp.copyto(phi_inserted_hat, phi_ref_hat_model, where=self.spec_filter_bool_m)
        return phi_m_obs_hat, phi_ref_obs_hat, phi_inserted_hat

    def _step_cda(self):
        """Apply continuous data assimilation nudging."""
        phi_m_hat = self._get_obs_hat(self.m_cda)
        phi_ref_hat = self._get_obs_hat(self.m_ref)

        if self.interpolant == 'spec':
            self.Ih_m, self.Ih_m_obs_hat = self._spec_intp(
                phi_m_hat, self.m_cda, self.spec_filter_mask_m
            )
            self.Ih_ref, self.Ih_ref_obs_hat = self._spec_intp(
                phi_ref_hat, self.m_ref, self.spec_filter_mask_ref
            )
            self.Ih_m_q_hat = self._obs_hat_to_q_hat(self.Ih_m_obs_hat)
            self.Ih_ref_q_hat = self._obs_hat_to_q_hat(self.Ih_ref_obs_hat)
            self.cda_forcing = self.mu * (self.Ih_ref_q_hat - self.Ih_m_q_hat)
            self.m_cda.da_term = self.cda_forcing
        elif self.interpolant == 'zeropad':
            phi_m_sub, phi_ref_sub = self._donwsampling(phi_m_hat, phi_ref_hat)
            self.Ih_m, self.Ih_m_obs_hat = self._zeropad_intp(phi_m_sub)
            self.Ih_ref, self.Ih_ref_obs_hat = self._zeropad_intp(phi_ref_sub)
            self.Ih_m_q_hat = self._obs_hat_to_q_hat(self.Ih_m_obs_hat)
            self.Ih_ref_q_hat = self._obs_hat_to_q_hat(self.Ih_ref_obs_hat)
            self.cda_forcing = self.mu * (self.Ih_ref_q_hat - self.Ih_m_q_hat)
            self.m_cda.da_term = self.cda_forcing
        elif self.interpolant == 'ot2003':
            (
                self.Ih_m_obs_hat,
                self.Ih_ref_obs_hat,
                self.Ih_ot_obs_hat,
            ) = self._ot2003_insert(phi_m_hat, phi_ref_hat)
            self.Ih_m_q_hat = self._obs_hat_to_q_hat(self.Ih_m_obs_hat)
            self.Ih_ref_q_hat = self._obs_hat_to_q_hat(self.Ih_ref_obs_hat)
            self.m_cda.q_hat = self._obs_hat_to_q_hat(self.Ih_ot_obs_hat)
            self.m_cda.p_hat = self.m_cda.inversion * self.m_cda.q_hat
            if self.m_cda.gamma:
                self.m_cda.rv_hat = (self.m_cda.q_hat
                                     + self.m_cda.gamma**2 * self.m_cda.p_hat)
            else:
                self.m_cda.rv_hat = self.m_cda.q_hat
            self.m_cda.da_term[...] = 0.0
            self.cda_forcing = self.m_cda.da_term
        else:
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

    def _apply_obs_spectral_filter(self, phi):
        """Low-pass filter a mapped observation field on the model grid."""
        if self.obs_filter_mask is None:
            return phi
        phi_hat = fft2(phi)
        phi_hat *= self.obs_filter_mask
        phi_filt = ifft2(phi_hat).real
        return phi_filt - cp.mean(phi_filt)

    def _linear_intp(self, phi):
        """Linear interpolation from observation grid to model grid"""
        phi_intp = map_coordinates(phi, self.rcoord_model, order=1, mode='grid-wrap', prefilter=False)
        phi_intp = phi_intp.reshape(self.m.Ny, self.m.Nx)
        return self._apply_obs_spectral_filter(phi_intp)

    def _block_intp(self, phi):
        """Block/nearest-neighbor interpolation from observation grid to model grid"""
        phi_intp = map_coordinates(phi, self.rcoord_model, order=0, mode='grid-wrap', prefilter=False)
        phi_intp = phi_intp.reshape(self.m.Ny, self.m.Nx)
        return self._apply_obs_spectral_filter(phi_intp)

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

    def _shell_sum(self, model, dens):
        """Complete-radial-shell sum using the CLE spatial-average convention."""
        key = id(model)
        if key not in self._shell_cache:
            self._shell_cache[key] = (
                model.kk_idx.ravel().get(),
                model.kk_range.get(),
                1.0 / (model.Nx * model.Ny) ** 2,
            )
        kk_idx_cpu, kk_sel, norm_fac = self._shell_cache[key]
        return npg.aggregate(kk_idx_cpu, dens.ravel().get(),
                             func='sum')[kk_sel] * norm_fac

    def _ref_q_hat_on_model(self, model):
        return self._to_mspace(self.m_ref.q_hat, self.m_ref, model)

    def _delta_hats(self, model):
        q_ref_hat = self._ref_q_hat_on_model(model)
        dq_hat = model.q_hat - q_ref_hat
        dpsi_hat = model.inversion * dq_hat
        return dq_hat, dpsi_hat, q_ref_hat

    def _finite_error_totals(self, model):
        """Finite error energy/enstrophy totals using CLE spatial averages."""
        dq_hat, dpsi_hat, _ = self._delta_hats(model)
        enorm = self._enorm(model, dpsi_hat)
        znorm = self._znorm(model, dq_hat)
        return enorm**2, znorm**2

    def _relative_error_spectra(self, model):
        """Old eerrk/zerrk diagnostics, renamed rdek/rdzk."""
        p_ref_hat = self._to_mspace(self.m_ref.p_hat, self.m_ref, model)
        q_ref_hat = self._to_mspace(self.m_ref.q_hat, self.m_ref, model)
        ek_model = np.asarray(model.get_Ek(model.p_hat))
        ek_ref = np.asarray(model.get_Ek(p_ref_hat))
        dek = np.asarray(model.get_Ek(model.p_hat - p_ref_hat))
        zk_model = np.asarray(model.get_Zk(model.q_hat))
        zk_ref = np.asarray(model.get_Zk(q_ref_hat))
        dzk = np.asarray(model.get_Zk(model.q_hat - q_ref_hat))
        rdek = dek / np.maximum(0.5 * (ek_model + ek_ref), self.errspec_eps)
        rdzk = dzk / np.maximum(0.5 * (zk_model + zk_ref), self.errspec_eps)
        return rdek, rdzk

    def _enorm(self, model, dpsi_hat):
        ene_dens = 0.5 * (model.kk**2 + model.gamma**2) * cp.abs(dpsi_hat)**2
        norm_fac = 1.0 / (model.Nx * model.Ny) ** 2
        ene = norm_fac * float(cp.sum(ene_dens, dtype=cp.float64).get())
        return float(np.sqrt(max(ene, 0.0)))

    def _znorm(self, model, dq_hat):
        ens_dens = 0.5 * cp.abs(dq_hat)**2
        norm_fac = 1.0 / (model.Nx * model.Ny) ** 2
        ens = norm_fac * float(cp.sum(ens_dens, dtype=cp.float64).get())
        return float(np.sqrt(max(ens, 0.0)))

    def _err_budget(self, model):
        """Energy-normalized finite-error E/Z/P spectra and budget terms."""
        dq_hat, dpsi_hat, q_ref_hat = self._delta_hats(model)
        enorm = self._enorm(model, dpsi_hat)
        epsi = dpsi_hat / max(enorm, 1e-300)
        eq = (model.lap - model.gamma**2) * epsi
        erv = eq + model.gamma**2 * epsi
        p_ref_hat = model.inversion * q_ref_hat
        k2 = model.kk**2

        evk = self._shell_sum(
            model, 0.5 * (k2 + model.gamma**2) * cp.abs(epsi)**2)
        zvk = self._shell_sum(model, 0.5 * cp.abs(eq)**2)
        pvk = self._shell_sum(model, 0.5 * k2 * cp.abs(eq)**2)

        j_adv = model._compute_jacobian(p_ref_hat, eq)
        j_prod = model._compute_jacobian(epsi, q_ref_hat)
        # Finite-amplitude correction to the linear CLE budget. Since epsi/eq
        # are normalized by ||delta psi||_E, the quadratic error interaction
        # enters the normalized error equation multiplied by the raw error norm.
        j_finite = enorm * model._compute_jacobian(epsi, eq)

        teadv = cp.real(cp.conj(epsi) * j_adv)
        teprod = cp.real(cp.conj(epsi) * j_prod)
        tefinite = cp.real(cp.conj(epsi) * j_finite)
        tzadv = -cp.real(cp.conj(eq) * j_adv)
        tzprod = -cp.real(cp.conj(eq) * j_prod)
        tzfinite = -cp.real(cp.conj(eq) * j_finite)
        tpadv = k2 * tzadv
        tpprod = k2 * tzprod
        tpfinite = k2 * tzfinite

        teadvk = self._shell_sum(model, teadv)
        teprodk = self._shell_sum(model, teprod)
        tefinitek = self._shell_sum(model, tefinite)
        tzadvk = self._shell_sum(model, tzadv)
        tzprodk = self._shell_sum(model, tzprod)
        tzfinitek = self._shell_sum(model, tzfinite)
        tpadvk = self._shell_sum(model, tpadv)
        tpprodk = self._shell_sum(model, tpprod)
        tpfinitek = self._shell_sum(model, tpfinite)

        fric_term = -model.friction_mask * model.friction * erv
        visc_term = model.hylap * erv
        tefric = -cp.real(cp.conj(epsi) * fric_term)
        tzfric = cp.real(cp.conj(eq) * fric_term)
        tevisc = -cp.real(cp.conj(epsi) * visc_term)
        tzvisc = cp.real(cp.conj(eq) * visc_term)
        tpvisc = k2 * tzvisc

        tefrick = self._shell_sum(model, tefric)
        tzfrick = self._shell_sum(model, tzfric)
        tevisck = self._shell_sum(model, tevisc)
        tzvisck = self._shell_sum(model, tzvisc)
        tpvisck = self._shell_sum(model, tpvisc)

        # DA acts in the raw q equation. Divide it by the raw error energy norm
        # before contracting it with the energy-normalized error fields.
        da_err_term = model.da_term / max(enorm, 1e-300)
        teda_err = -cp.real(cp.conj(epsi) * da_err_term)
        tzda_err = cp.real(cp.conj(eq) * da_err_term)
        tpda_err = k2 * tzda_err
        tedafk = self._shell_sum(model, teda_err)
        tzdafk = self._shell_sum(model, tzda_err)
        tpdafk = self._shell_sum(model, tpda_err)

        return (evk, zvk, pvk, teadvk, teprodk, tefinitek, tefrick,
                tevisck, tzadvk, tzprodk, tzfinitek, tzfrick, tzvisck,
                tpadvk, tpprodk, tpfinitek, tpvisck, tedafk,
                tzdafk, tpdafk)

    def _bind_var(self, ds, name, dtype, dims, description=None):
        if name in ds.variables:
            var = ds.variables[name]
        else:
            var = ds.createVariable(name, dtype, dims)
        if description is not None:
            var.description = description
        return var

    def _bind_error_diag_vars(self, model):
        diag = {}
        diag['de'] = self._bind_var(
            model.ds, 'de', 'f8', ('time',),
            'finite error energy, spatial-mean Parseval total over full FFT square')
        diag['dz'] = self._bind_var(
            model.ds, 'dz', 'f8', ('time',),
            'finite error enstrophy, spatial-mean Parseval total over full FFT square')
        for name in ('evk', 'zvk', 'pvk', 'teadvk', 'teprodk', 'tefinitek',
                     'tefrick', 'tevisck', 'tzadvk', 'tzprodk',
                     'tzfinitek', 'tzfrick', 'tzvisck', 'tpadvk',
                     'tpprodk', 'tpfinitek', 'tpvisck', 'tedafk',
                     'tzdafk', 'tpdafk', 'tprodk', 'tdissk',
                     'rdek', 'rdzk'):
            diag[name] = self._bind_var(model.ds, name, 'f4', ('time', 'k'))
        diag['evk'].description = ('full-energy-normalized error spectrum on complete '
                                   'radial shells k < N/2')
        diag['zvk'].description = ('full-energy-normalized enstrophy spectrum on complete '
                                   'radial shells k < N/2')
        diag['pvk'].description = ('full-energy-normalized error palinstrophy spectrum '
                                   '0.5 k^2 |delta q|^2 on complete radial shells k < N/2')
        diag['rdek'].description = 'relative finite energy-error spectrum, old eerrk'
        diag['rdzk'].description = 'relative finite enstrophy-error spectrum, old zerrk'
        diag['tefinitek'].description = 'finite-amplitude nonlinear energy error budget term'
        diag['tzfinitek'].description = 'finite-amplitude nonlinear enstrophy error budget term'
        diag['tpadvk'].description = ('error palinstrophy tendency from reference-flow '
                                     'advection on complete radial shells k < N/2')
        diag['tpprodk'].description = ('error palinstrophy tendency from perturbation '
                                      'advection of reference q on complete radial shells k < N/2')
        diag['tpfinitek'].description = ('finite-amplitude nonlinear error palinstrophy '
                                        'budget term on complete radial shells k < N/2')
        diag['tpvisck'].description = ('viscous error palinstrophy tendency on complete '
                                      'radial shells k < N/2; for gamma=0/all-mode drag, '
                                      'the omitted drag tendency is -2*friction*pvk')
        diag['tedafk'].description = (
            'normalized finite-error energy tendency from the instantaneous DA RHS coupling; '
            'zero for impulsive OT2003 insertion, whose saved state is post-insertion')
        diag['tzdafk'].description = (
            'normalized finite-error enstrophy tendency from the instantaneous DA RHS '
            'coupling; zero for impulsive OT2003 insertion, whose saved state is post-insertion')
        diag['tpdafk'].description = (
            'normalized finite-error palinstrophy tendency from the instantaneous DA RHS '
            'coupling; zero for impulsive OT2003 insertion, whose saved state is post-insertion')
        diag['tprodk'].description = 'combined energy production teadvk + teprodk + tefinitek'
        diag['tdissk'].description = 'combined energy dissipation tefrick + tevisck'
        return diag

    def _save_error_diag(self, model, diag, it):
        de, dz = self._finite_error_totals(model)
        rdek, rdzk = self._relative_error_spectra(model)
        (evk, zvk, pvk, teadvk, teprodk, tefinitek, tefrick, tevisck,
         tzadvk, tzprodk, tzfinitek, tzfrick, tzvisck, tpadvk,
         tpprodk, tpfinitek, tpvisck, tedafk, tzdafk,
         tpdafk) = self._err_budget(model)
        diag['de'][it] = de
        diag['dz'][it] = dz
        diag['evk'][it, :] = evk
        diag['zvk'][it, :] = zvk
        diag['pvk'][it, :] = pvk
        diag['rdek'][it, :] = rdek
        diag['rdzk'][it, :] = rdzk
        diag['teadvk'][it, :] = teadvk
        diag['teprodk'][it, :] = teprodk
        diag['tefinitek'][it, :] = tefinitek
        diag['tefrick'][it, :] = tefrick
        diag['tevisck'][it, :] = tevisck
        diag['tzadvk'][it, :] = tzadvk
        diag['tzprodk'][it, :] = tzprodk
        diag['tzfinitek'][it, :] = tzfinitek
        diag['tzfrick'][it, :] = tzfrick
        diag['tzvisck'][it, :] = tzvisck
        diag['tpadvk'][it, :] = tpadvk
        diag['tpprodk'][it, :] = tpprodk
        diag['tpfinitek'][it, :] = tpfinitek
        diag['tpvisck'][it, :] = tpvisck
        diag['tedafk'][it, :] = tedafk
        diag['tzdafk'][it, :] = tzdafk
        diag['tpdafk'][it, :] = tpdafk
        diag['tprodk'][it, :] = teadvk + teprodk + tefinitek
        diag['tdissk'][it, :] = tefrick + tevisck

    def _set_cda_output_attrs(self, model, role):
        model.ds.Nobs = self.Nobs
        model.ds.interpolant = self.interpolant
        model.ds.mu = self.mu
        model.ds.obs_field = self.obs_field
        model.ds.dTobs = self.dTobs
        model.ds.trajectory_role = role

    def create_ctrl_nc(self, nf):
        if not self.is_ctrl:
            return
        self.m.create_nc(nf, prefix='ctrl_o')
        self._set_cda_output_attrs(self.m, 'control')
        self.errdiag_ctrl_vars = self._bind_error_diag_vars(self.m)

    def create_cda_nc(self, nf):
        self.m_cda.create_nc(nf, prefix='cda_o')
        self._set_cda_output_attrs(self.m_cda, 'assimilated')
        self.errdiag_cda_vars = self._bind_error_diag_vars(self.m_cda)
        if 'Ihm' not in self.m_cda.ds.variables:
            self.Ih_m_var = self.m_cda.ds.createVariable('Ihm', 'f8', ('time', 'y', 'x'))
        else:
            self.Ih_m_var = self.m_cda.ds.variables['Ihm']
        if 'Ihref' not in self.m_cda.ds.variables:
            self.Ih_ref_var = self.m_cda.ds.createVariable('Ihref', 'f8', ('time', 'y', 'x'))
        else:
            self.Ih_ref_var = self.m_cda.ds.variables['Ihref']
        self.Ih_m_var.description = ('observation operator applied to the CDA state, '
                                     f'in {self.obs_field} space')
        self.Ih_ref_var.description = ('observation operator applied to the reference state, '
                                       f'in {self.obs_field} space')

    def create_gnud_nc(self, nf):
        if self.is_gnuding:
            self.m_gnud.create_nc(nf,prefix='gnud_o')
            self._set_cda_output_attrs(self.m_gnud, 'grid_nudging')
            self.errdiag_gnud_vars = self._bind_error_diag_vars(self.m_gnud)

    def create_ref_nc(self, nf):
        self.m_ref.create_nc(nf, prefix='ref_o')

    def create_nc(self,nf):
        self.create_ctrl_nc(nf)
        self.create_cda_nc(nf)
        self.create_gnud_nc(nf)
        self.create_ref_nc(nf)

    def create_ctrl_rst(self, nf):
        if not self.is_ctrl:
            return
        self.m.create_rst(nf, prefix='ctrl_r') 
        self.tcda_ctrl_var = self._bind_tcda_var(self.m.rstds)

    def _bind_tcda_var(self, ds):
        if 'tcda' not in ds.variables:
            return ds.createVariable('tcda', 'f8', ('time',))
        else:
            return ds.variables['tcda']

    def create_cda_rst(self, nf):     
        self.m_cda.create_rst(nf, prefix='cda_r')
        self.tcda_cda_var = self._bind_tcda_var(self.m_cda.rstds)
        
    def create_gnud_rst(self, nf):
        if self.is_gnuding:
            self.m_gnud.create_rst(nf, prefix='gnud_r')
            
    def create_ref_rst(self, nf):
        self.m_ref.create_rst(nf, prefix='ref_r')
        
    def create_rst(self, nf):
        self.create_ctrl_rst(nf)
        self.create_cda_rst(nf)
        self.create_gnud_rst(nf)
        self.create_ref_rst(nf)

    def close_nc(self):
        if self.is_ctrl:
            self.m.ds.close()
        self.m_cda.ds.close()
        if self.is_gnuding:
            self.m_gnud.ds.close()
        self.m_ref.ds.close()

    def close_rst(self):
        if self.is_ctrl:
            self.m.rstds.close()
        self.m_cda.rstds.close()
        if self.is_gnuding:
            self.m_gnud.rstds.close()
        self.m_ref.rstds.close()

    def save_var(self, it):
        """Save variables to netCDF output"""
        if self.is_ctrl:
            self.m.save_var(it)
            self._save_error_diag(self.m, self.errdiag_ctrl_vars, it)
            self.m.ds.sync()

        self.m_cda.save_var(it)
        self._save_error_diag(self.m_cda, self.errdiag_cda_vars, it)
        if self.interpolant == 'ot2003' and hasattr(self, 'Ih_m_obs_hat'):
            self.Ih_m = ifft2(self.Ih_m_obs_hat).real
            self.Ih_ref = ifft2(self.Ih_ref_obs_hat).real
            self.Ih_m -= cp.mean(self.Ih_m)
            self.Ih_ref -= cp.mean(self.Ih_ref)
        self.Ih_m_var[it, :, :] = self.Ih_m.get()
        self.Ih_ref_var[it, :, :] = self.Ih_ref.get()
        self.m_cda.ds.sync()

        if self.is_gnuding:
            self.m_gnud.save_var(it)
            self._save_error_diag(self.m_gnud, self.errdiag_gnud_vars, it)
            self.m_gnud.ds.sync()

        self.m_ref.save_var(it)
        self.m_ref.ds.sync()
    def save_rst(self,it):
        """Save restart files"""
        if self.is_ctrl:
            self.m.save_rst(it)
            self.tcda_ctrl_var[it] = self.rt
            self.m.rstds.sync()
        self.m_cda.save_rst(it)
        self.tcda_cda_var[it] = self.rt
        self.m_cda.rstds.sync()
        if self.is_gnuding:
            self.m_gnud.save_rst(it)
        self.m_ref.save_rst(it)

    def _set_model_times(self, rt):
        """Set absolute model time = pickup time from spinup + CDA relative time."""
        if self.is_ctrl:
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
            saveplot: Ignored compatibility flag; CDA diagnostics are saved to NetCDF only.
        """
        if self.is_ctrl:
            self.m.ts_scheme = scheme
        self.m_cda.ts_scheme = scheme
        if self.is_gnuding:
            self.m_gnud.ts_scheme = scheme
        self.m_ref.ts_scheme = scheme

        if self.is_ctrl:
            self.m.savedir = savedir
        self.m_cda.savedir = savedir
        if self.is_gnuding:
            self.m_gnud.savedir = savedir
        self.m_ref.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        self.tsave_rst = tsave_rst

        total_steps = int(round(tmax/self.dt))

        print(f"Starting CDA. dTobs={self.dTobs}, tmax={tmax}")
        print(f"Step interval -> CDA: {self.intvl_model}, Ref: {self.intvl_ref}, Obs: {self.intvl_da}")

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
            if self.is_ctrl:
                self.m.n_steps = n
            self.m_cda.n_steps = n
            if self.is_gnuding:
                self.m_gnud.n_steps = n
            self.m_ref.n_steps = n
            is_da_step = n % self.intvl_da == 0
            # print diagnostics to console every 10,000 steps
            if n%10000 == 0:
                # Compute and print energy and diagnostic statistics
                E_cda = self.m_cda.get_Etot(self.m_cda.p_hat)/self.m_cda.Nx/self.m_cda.Ny
                E_ref = self.m_ref.get_Etot(self.m_ref.p_hat)/self.m_ref.Nx/self.m_ref.Ny
                diag_parts = [
                    f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
                    f"step {self.m_cda.n_steps:7d}",
                    f"t={self.m_cda.t:9.6f}s",
                ]
                if self.is_ctrl:
                    E_ctrl = self.m.get_Etot(self.m.p_hat)/self.m.Nx/self.m.Ny
                    diag_parts.append(f"E_ctrl={E_ctrl:.4e}")
                diag_parts.append(f"E_cda={E_cda:.4e}")
                if self.is_gnuding:
                    E_gnud = self.m_gnud.get_Etot(self.m_gnud.p_hat)/self.m_gnud.Nx/self.m_gnud.Ny
                    diag_parts.append(f"E_gnud={E_gnud:.4e}")
                diag_parts.append(f"E_ref={E_ref:.4e}")
                print("      ".join(diag_parts), end="\n")

            # Refresh the observation coupling before any coincident output.
            # OT2003 output is therefore post-insertion, while continuous
            # nudging output contains the DA term used by the next model step.
            if is_da_step:
                self._step_cda()
                if self.is_gnuding:
                    self._step_gnud()

            if n % self.intvl_model ==0:
                if self.m_cda.n_steps % tsave == 0:

                    if insave == nsave:
                        itsave =0 #time index -it
                        if nf > nf0 : 
                            self.close_nc()
                        self.create_nc(nf)
                        insave=0 #save number index-in
                        nf+=1
                
                    self.save_var(itsave)
                    print(f"[save_var] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.m_cda.n_steps:7d}  t={self.m_cda.t:9.6f}s ", end="\n")
                
                    
                    itsave +=1
                    insave +=1  
                if self.m_cda.n_steps % tsave_rst==0:
                    if inrst == nrst:
                        itrst =0 #time index -it
                        if nfrst > nfrst0 : 
                            self.close_rst()
                        self.create_rst(nfrst)
                        inrst=0 #save number index-in
                        nfrst+=1

                    self.save_rst(itrst)
                    print(f"[save_rst] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.m_cda.n_steps:7d}  t={self.m_cda.t:9.6f}s ", end="\n")
                
                    itrst +=1
                    inrst +=1

                if self.is_ctrl:
                    self.m._step_forward()
                self.m_cda._step_forward() 
                if self.is_gnuding:
                    self.m_gnud._step_forward()
                self.m_cda.da_term[...] = 0.0
                if self.is_gnuding:
                    self.m_gnud.da_term[...] = 0.0
                # Reconstruct absolute model clocks from integer step index.
                # This avoids cumulative floating-point drift from repeated += dt.
                if self.is_ctrl:
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
