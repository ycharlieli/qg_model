import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from scipy.fft import fft2, ifft2
from scipy.fft import set_global_backend
set_global_backend(cufft)
import numpy_groupies as npg
import netCDF4 as nc
import os
import copy
import time


class QGCLE:
    """Conditional Lyapunov exponent and conditional LLV for QGModel.

    Master-slave setup mimicking cda_turb2d.py with the OT2003 exact spectral
    insertion: at every dTobs the slave's observed modes (|k| < Nobs) are
    replaced by the master's, so the error lives entirely in
    the unobserved subspace (Li et al. 2025b, eq. 2.5). The slave starts
    epsilon-close to the master and the error
    is rescaled back to a fixed small energy norm every dT_cle, the renormalized
    two-trajectory
    method of Boffetta & Musacchio (2017) / Li et al. (2024), so both signs of
    the CLE are measurable indefinitely.

    Every dT_cle the module records finite-interval logarithmic growth rates
        lam_i   = ln(||delta||_E / ||delta||_E,prev) / dT_cle
        lam_i_z = ln(||delta||_Z / ||delta||_Z,prev) / dT_cle
    (||.||_E the energy norm, the fixed rescaling norm matching the velocity
    2-norm used by Li; ||.||_Z the enstrophy norm, i.e. the L2 norm of
    the state q), the running averages lam/lam_z (both estimate the CLE), the
    error norms znorm and enorm (synchronisation error norms of the
    literature, not RMSE), the normalized error (conditional LLV) energy,
    enstrophy and PV-gradient palinstrophy spectra, and their exact linearized
    spectral-budget terms following Li et al. (2025b), adapted to the QG PV
    equation. No cascade, principal-strain or alignment mechanism is assumed.
    The diagnostics also save Li-style combined production
    tprodk = teadvk + teprodk and scalar running averages of production,
    dissipation and prod - diss for comparison with lam. Both velocity-based
    quantities (evk, te*) and q-based diagnostic quantities (zvk, pvk, tz*,
    tp*) are evaluated from the same energy-normalized conditional LLV; the
    q-side diagnostics are not separately normalized to unit enstrophy or
    palinstrophy. Palinstrophy densities are weighted by each Fourier mode's
    exact k^2 before shell aggregation. For the target gamma=0, all-mode-drag
    runs, the omitted drag tendency is exactly -2*friction*pvk mode by mode;
    no redundant combined or friction palinstrophy arrays are stored.
    Budget spectra are instantaneous endpoint diagnostics at the save time;
    they are not interval averages of lam_i.

    Snapshots save q of the master, raw delta_psi and delta_q, and the
    energy-normalized streamfunction error field
    llv = delta_psi/||delta||_E; no flux diagnostics, Ihm/Ihref or daF are
    saved.
    """

    def __init__(self, m_ref=None, m_cle=None, Nobs=16, dTobs=None, dT_cle=0.1,
                 epsilon_rel=1e-4, epsilon_abs=None, seed=10,
                 is_not_rst=True, rtrst=0.0):
        """Args:
            m_ref: Master (truth) QGModel with initial condition already set.
            m_cle: Optional slave (pass when resuming from restart files). If
                None, it is created as deepcopy(m_ref) plus a random
                streamfunction perturbation confined to modes |k| >= Nobs with
                energy norm epsilon.
            Nobs: Observation cutoff wavenumber (modes with |k| < Nobs are
                inserted from the master).
            dTobs: Insertion interval; defaults to the model time step.
            dT_cle: Rescaling interval; also the diagnostic save cadence.
            epsilon_rel: Perturbation energy norm relative to the master state.
            epsilon_abs: Absolute perturbation energy norm (overrides rel).
            seed: CuPy RNG seed for the initial perturbation.
            is_not_rst: False to resume a previous run at time rtrst.
            rtrst: Relative resume time (multiple of dT_cle and tsave_rst*dt).
        """
        self.m_ref = m_ref
        self.dt = self.m_ref.dt
        self.Nobs = Nobs
        self.dTobs = float(dTobs) if dTobs is not None else self.dt
        self.intvl_da = int(round(self.dTobs / self.dt))
        self.dT = float(dT_cle)
        self.intvl = int(round(self.dT / self.dt))
        if self.intvl % self.intvl_da != 0:
            raise ValueError("dT_cle must be a multiple of dTobs so rescaling "
                             "happens right after an insertion.")
        self.is_not_rst = is_not_rst
        self.rtrst = rtrst
        self.seed = seed

        k_radius = cp.sqrt(m_ref.nx2d.astype(cp.float64)**2
                           + m_ref.ny2d.astype(cp.float64)**2)
        self.obs_mask = (k_radius < float(Nobs)).astype(m_ref.rdtype)
        self.obs_bool = self.obs_mask.astype(cp.bool_)
        self.unobs_mask = 1.0 - self.obs_mask
        self.unobs_mask[0, 0] = 0.0

        # Shell spectra use complete radial shells; scalar reductions below
        # use the full FFT square.
        self._kk_idx_cpu = m_ref.kk_idx.ravel().get()
        self._kk_sel = m_ref.kk_range.get()
        # Parseval with the unnormalized FFT needs 1/(Nx*Ny)^2 so norms and
        # spectra are spatial averages (resolution-independent, comparable
        # across N and with the literature)
        self._norm_fac = 1.0 / (m_ref.Nx * m_ref.Ny) ** 2

        base_norm = self._enorm(m_ref.p_hat)
        if epsilon_abs is not None:
            self.epsilon = float(epsilon_abs)
        else:
            self.epsilon = float(epsilon_rel) * max(base_norm, 1e-30)
        if self.epsilon <= 0.0:
            raise ValueError(f"Perturbation norm must be positive, got {self.epsilon}.")

        if m_cle is not None:
            if m_cle.precision != m_ref.precision:
                raise ValueError(
                    f"Master and slave precisions differ: {m_ref.precision!r} "
                    f"vs {m_cle.precision!r}. Build both with the same precision.")
            self.m_cle = m_cle
        else:
            self.m_cle = copy.deepcopy(m_ref)
            cp.random.seed(seed)
            self.m_cle.q_hat = self.m_cle.q_hat + self._make_random_delta(self.unobs_mask)
            self._sync_state(self.m_cle)

    # ---------------- perturbation and coupling helpers ----------------
    def _make_random_delta(self, mask):
        noise = cp.random.randn(self.m_ref.Ny, self.m_ref.Nx).astype(self.m_ref.rdtype)
        delta_psi_hat = fft2(noise).astype(self.m_ref.cdtype)
        delta_psi_hat *= mask
        delta_psi_hat[0, 0] = 0.0
        self.m_ref._enforce_spectral_constraints(delta_psi_hat)
        norm = self._enorm(delta_psi_hat)
        if norm <= 0.0:
            raise RuntimeError("Random perturbation has zero energy norm after masking.")
        return self._psi_to_q((self.epsilon / norm) * delta_psi_hat)

    def _sync_state(self, model):
        model._enforce_spectral_constraints(model.q_hat)
        model.p_hat = model.inversion * model.q_hat
        if model.gamma:
            model.rv_hat = model.q_hat + model.gamma**2 * model.p_hat
        else:
            model.rv_hat = model.q_hat

    def _insert(self, sync_state=True):
        """OT2003 exact insertion of the observed modes of the master."""
        cp.copyto(self.m_cle.q_hat, self.m_ref.q_hat, where=self.obs_bool)
        if sync_state:
            self._sync_state(self.m_cle)

    # ---------------- norms and spectra ----------------
    def _psi_to_q(self, dpsi_hat):
        return (self.m_ref.lap - self.m_ref.gamma**2) * dpsi_hat

    def _q_to_psi(self, dq_hat):
        return self.m_ref.inversion * dq_hat

    def _enorm(self, dpsi_hat):
        """Energy norm of a streamfunction perturbation."""
        ene_dens = 0.5 * (self.m_ref.kk**2 + self.m_ref.gamma**2) * cp.abs(dpsi_hat)**2
        ene = self._norm_fac * float(cp.sum(ene_dens, dtype=cp.float64).get())
        return float(np.sqrt(max(ene, 0.0)))

    def _znorm(self, dq_hat):
        """Enstrophy norm of a PV perturbation."""
        ens_dens = 0.5 * cp.abs(dq_hat)**2
        ens = self._norm_fac * float(cp.sum(ens_dens, dtype=cp.float64).get())
        return float(np.sqrt(max(ens, 0.0)))

    def _shell_sum(self, dens):
        return npg.aggregate(self._kk_idx_cpu, dens.ravel().get(),
                             func='sum')[self._kk_sel] * self._norm_fac

    def _delta_q(self):
        return (self.m_cle.q_hat - self.m_ref.q_hat) * self.unobs_mask

    def _delta_psi(self, dq_hat=None):
        if dq_hat is None:
            dq_hat = self._delta_q()
        return self._q_to_psi(dq_hat)

    def _err_budget(self, dpsi_hat, enorm):
        """Spectra of the normalized error (conditional LLV) and its energy,
        enstrophy and PV-gradient palinstrophy budget terms, linearized about
        the master trajectory (Li et al. 2025b, eqs. 2.27--2.29, adapted to
        the QG PV equation). Sign conventions follow
        get_TENL/get_diagFric/get_diagVisc of turb2d.py."""
        m = self.m_ref
        # Energy-normalized LLV in streamfunction form; q/rv are derived from it
        # only because the QG equation is written for PV.
        epsi = dpsi_hat / max(enorm, 1e-300)
        eq = self._psi_to_q(epsi)
        erv = eq + m.gamma**2 * epsi

        # Normalized error spectra (energy spectrum = conditional LLV spectrum).
        # Apply the exact modal k^2 before shell aggregation: rounded shell
        # centres are not exact substitutes for the modal wavenumbers.
        evk = self._shell_sum(0.5 * (m.kk**2 + m.gamma**2) * cp.abs(epsi)**2)
        zvk = self._shell_sum(0.5 * cp.abs(eq)**2)
        pvk = self._shell_sum(0.5 * m.kk**2 * cp.abs(eq)**2)

        # advective transfer J(psi_ref, dq) and production J(dpsi, q_ref)
        j_adv = m._compute_jacobian(m.p_hat, eq)
        j_prod = m._compute_jacobian(epsi, m.q_hat)
        teadv = cp.real(cp.conj(epsi) * j_adv)
        teprod = cp.real(cp.conj(epsi) * j_prod)
        tzadv = -cp.real(cp.conj(eq) * j_adv)
        tzprod = -cp.real(cp.conj(eq) * j_prod)
        tpadv = m.kk**2 * tzadv
        tpprod = m.kk**2 * tzprod
        teadvk = self._shell_sum(teadv)
        teprodk = self._shell_sum(teprod)
        tzadvk = self._shell_sum(tzadv)
        tzprodk = self._shell_sum(tzprod)
        tpadvk = self._shell_sum(tpadv)
        tpprodk = self._shell_sum(tpprod)

        # linear dissipation acting on the error (beta term is energy-neutral;
        # forcing is identical in both runs and cancels; Leith and the Arbic
        # filter are not included in this budget)
        fric_term = -m.friction_mask * m.friction * erv
        visc_term = m.hylap * erv
        tefric = -cp.real(cp.conj(epsi) * fric_term)
        tzfric = cp.real(cp.conj(eq) * fric_term)
        tevisc = -cp.real(cp.conj(epsi) * visc_term)
        tzvisc = cp.real(cp.conj(eq) * visc_term)
        tpvisc = m.kk**2 * tzvisc
        # Target 2-D NSE runs have gamma=0 and all-mode drag, hence the
        # unsaved palinstrophy drag tendency is exactly -2*friction*pvk.
        tefrick = self._shell_sum(tefric)
        tzfrick = self._shell_sum(tzfric)
        tevisck = self._shell_sum(tevisc)
        tzvisck = self._shell_sum(tzvisc)
        tpvisck = self._shell_sum(tpvisc)

        # Integrated budget terms include every Fourier mode; the saved *k
        # arrays remain isotropic spectra on complete radial shells.
        prod_i = 0.5 * self._norm_fac * float(
            cp.sum(teadv + teprod, dtype=cp.float64).get())
        diss_i = -0.5 * self._norm_fac * float(
            cp.sum(tefric + tevisc, dtype=cp.float64).get())

        return (evk, zvk, pvk, teadvk, teprodk, tefrick, tevisck,
                tzadvk, tzprodk, tzfrick, tzvisck,
                tpadvk, tpprodk, tpvisck, prod_i, diss_i)

    def _herm_project(self, d_hat):
        """Project a spectral difference onto the Hermitian (real-field)
        subspace. FFT roundoff seeds an anti-Hermitian component that the
        .real-based Jacobian cannot see; without this projection repeated
        rescaling amplifies it into a spurious lam = -(nu k_eff^2 + alpha)
        mode whenever the physical CLE is more stable than that."""
        return fft2(ifft2(d_hat).real)

    def _rescale(self, fac):
        """Pull the slave back to distance epsilon from the master."""
        self.m_cle.q_hat = self.m_ref.q_hat + fac * self._herm_project(
            self.m_cle.q_hat - self.m_ref.q_hat)
        self.m_cle.k1_p = self.m_ref.k1_p + fac * self._herm_project(
            self.m_cle.k1_p - self.m_ref.k1_p)
        self.m_cle.k1_pp = self.m_ref.k1_pp + fac * self._herm_project(
            self.m_cle.k1_pp - self.m_ref.k1_pp)
        self._sync_state(self.m_cle)

    # ---------------- output ----------------
    def _bind_diag_var(self, name, dtype, dims, description):
        if name in self.dds.variables:
            var = self.dds.variables[name]
        else:
            var = self.dds.createVariable(name, dtype, dims)
        var.description = description
        return var

    def _bind_diag_vars(self):
        self.d_times = self.dds.variables['time']
        self.lami_var = self.dds.variables['lam_i']
        self.lam_var = self.dds.variables['lam']
        self.enorm_var = self.dds.variables['enorm']
        self.lamiz_var = self.dds.variables['lam_i_z']
        self.lamz_var = self.dds.variables['lam_z']
        self.znorm_var = self.dds.variables['znorm']
        self.evk_var = self.dds.variables['evk']
        self.zvk_var = self.dds.variables['zvk']
        self.pvk_var = self._bind_diag_var(
            'pvk', 'f4', ('time', 'k'),
            'energy-normalized LLV PV-gradient palinstrophy spectrum; '
            'modal density 0.5*k_mode^2*|eq|^2 is weighted before the '
            'complete-radial-shell sum')
        self.teadvk_var = self.dds.variables['teadvk']
        self.teprodk_var = self.dds.variables['teprodk']
        self.tefrick_var = self.dds.variables['tefrick']
        self.tevisck_var = self.dds.variables['tevisck']
        self.tzadvk_var = self.dds.variables['tzadvk']
        self.tzprodk_var = self.dds.variables['tzprodk']
        self.tzfrick_var = self.dds.variables['tzfrick']
        self.tzvisck_var = self.dds.variables['tzvisck']
        self.tpadvk_var = self._bind_diag_var(
            'tpadvk', 'f4', ('time', 'k'),
            'signed advective tendency of energy-normalized LLV '
            'PV-gradient palinstrophy, '
            '-k_mode^2*Re[conj(eq)*J(psi_ref,eq)], weighted before the '
            'complete-radial-shell sum')
        self.tpprodk_var = self._bind_diag_var(
            'tpprodk', 'f4', ('time', 'k'),
            'signed production tendency of energy-normalized LLV '
            'PV-gradient palinstrophy, '
            '-k_mode^2*Re[conj(eq)*J(epsi,q_ref)], weighted before the '
            'complete-radial-shell sum')
        self.tpvisck_var = self._bind_diag_var(
            'tpvisck', 'f4', ('time', 'k'),
            'signed hyperviscous tendency of energy-normalized LLV '
            'PV-gradient palinstrophy, '
            'k_mode^2*Re[conj(eq)*(hylap*erv)], weighted before the '
            'complete-radial-shell sum')
        self.tprodk_var = self._bind_diag_var(
            'tprodk', 'f4', ('time', 'k'),
            'Li-style combined production spectrum teadvk + teprodk '
            'on complete radial shells k < N/2')
        self.tdissk_var = self._bind_diag_var(
            'tdissk', 'f4', ('time', 'k'),
            'combined energy dissipation tendency tefrick + tevisck '
            'on complete radial shells k < N/2')
        self.prodi_var = self._bind_diag_var(
            'prod_i', 'f8', ('time',),
            'instantaneous full-FFT-square production, comparable to Li production')
        self.dissi_var = self._bind_diag_var(
            'diss_i', 'f8', ('time',),
            'instantaneous full-FFT-square positive Li-style dissipation')
        self.lbudgeti_var = self._bind_diag_var(
            'lam_budget_i', 'f8', ('time',),
            'instantaneous prod_i - diss_i, comparable to local lambda')
        self.lbudget_residi_var = self._bind_diag_var(
            'lam_budget_resid_i', 'f8', ('time',),
            'instantaneous lam_i - lam_budget_i')
        self.prod_var = self._bind_diag_var(
            'prod', 'f8', ('time',), 'running mean of prod_i')
        self.diss_var = self._bind_diag_var(
            'diss', 'f8', ('time',), 'running mean of diss_i')
        self.lbudget_var = self._bind_diag_var(
            'lam_budget', 'f8', ('time',), 'running mean of lam_budget_i')
        self.lbudget_resid_var = self._bind_diag_var(
            'lam_budget_resid', 'f8', ('time',), 'running lam - lam_budget')

    def _stamp_output_metadata(self, ds):
        """Add model and spectral conventions to CLE diagnostic/snapshot files."""
        m = self.m_ref
        ds.beta = float(m.beta)
        ds.gamma = float(m.gamma)
        ds.hyvisc = float(m.hyvisc)
        ds.hyperorder = int(m.hyperorder)
        ds.friction = float(m.friction)
        ds.k_friction = float(m.k_friction)
        ds.kf = float(m.fscale)
        ds.cl = float(m.cl)
        ds.sp_filtr = int(bool(m.sp_filtr))
        ds.Nx = int(m.Nx)
        ds.Ny = int(m.Ny)
        ds.Lx = float(m.Lx)
        ds.Ly = float(m.Ly)
        ds.normalization = (
            'llv = delta_psi/||delta||_E, with ||delta||_E^2 = '
            '(Nx*Ny)^-2 sum_modes 0.5*(k^2+gamma^2)*|delta_psi_hat|^2; '
            'spectral arrays are (Nx*Ny)^-2 sums of their stated modal '
            'densities and all E/Z/P arrays use this energy-normalized llv'
        )
        ds.shell_convention = (
            'rounded radial grid-index shells; k = shell_index*(2*pi/Lx); '
            'save shell_index < Nx/2; every k-dependent density weight uses '
            'the exact modal wavenumber before shell aggregation'
        )

    def _adopt_diag_epsilon(self, ds, nc_filename):
        stored_eps = float(ds.epsilon)
        if not np.isclose(stored_eps, self.epsilon, rtol=1e-6):
            print(f"[create_diag_nc] adopting epsilon={stored_eps:.6e} from "
                  f"{nc_filename} (constructor gave {self.epsilon:.6e}) so the "
                  "rescaling target stays constant across restarts")
            self.epsilon = stored_eps

    def create_diag_nc(self, nf, prefix='cle_d'):
        nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
        if os.path.exists(nc_filename) and not self.is_not_rst:
            self.dds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            self._adopt_diag_epsilon(self.dds, nc_filename)
            self._bind_diag_vars()
        else:
            self.dds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            self.dds.createDimension('time', None)
            self.dds.createDimension('k', len(self.m_ref.kk_iso))
            kk = self.dds.createVariable('k', 'f4', ('k',))
            kk[:] = self.m_ref.kk_iso.get()
            self.d_times = self.dds.createVariable('time', 'f8', ('time',))
            self.lami_var = self.dds.createVariable('lam_i', 'f8', ('time',))
            self.lam_var = self.dds.createVariable('lam', 'f8', ('time',))
            self.enorm_var = self.dds.createVariable('enorm', 'f8', ('time',))
            self.lamiz_var = self.dds.createVariable('lam_i_z', 'f8', ('time',))
            self.lamz_var = self.dds.createVariable('lam_z', 'f8', ('time',))
            self.znorm_var = self.dds.createVariable('znorm', 'f8', ('time',))
            self.evk_var = self.dds.createVariable('evk', 'f4', ('time', 'k'))
            self.zvk_var = self.dds.createVariable('zvk', 'f4', ('time', 'k'))
            self.teadvk_var = self.dds.createVariable('teadvk', 'f4', ('time', 'k'))
            self.teprodk_var = self.dds.createVariable('teprodk', 'f4', ('time', 'k'))
            self.tefrick_var = self.dds.createVariable('tefrick', 'f4', ('time', 'k'))
            self.tevisck_var = self.dds.createVariable('tevisck', 'f4', ('time', 'k'))
            self.tzadvk_var = self.dds.createVariable('tzadvk', 'f4', ('time', 'k'))
            self.tzprodk_var = self.dds.createVariable('tzprodk', 'f4', ('time', 'k'))
            self.tzfrick_var = self.dds.createVariable('tzfrick', 'f4', ('time', 'k'))
            self.tzvisck_var = self.dds.createVariable('tzvisck', 'f4', ('time', 'k'))
            self._bind_diag_vars()
            self.dds.description = "QG CLE/conditional-LLV diagnostics (master-slave, rescaled)"
            self.dds.precision = self.m_ref.precision
            self.dds.Nobs = self.Nobs
            self.dds.dTobs = self.dTobs
            self.dds.dT_cle = self.dT
            self.dds.epsilon = self.epsilon
            self.dds.rescale_norm = "energy"
            self.dds.seed = self.seed
            self.dds.dt = self.dt
            self.dds.file_index = nf
        # Also stamp append-open legacy files. Variables created by
        # _bind_diag_vars above retain fill values at historical records and
        # are populated only at newly saved restart records.
        self._stamp_output_metadata(self.dds)

    def _init_lam_sums(self, n_diag_done, nsave, prefix='cle_d'):
        self._lam_sum = 0.0
        self._lam_n = 0
        self._lamz_sum = 0.0
        self._lamz_n = 0
        self._prod_sum = 0.0
        self._diss_sum = 0.0
        self._lbudget_sum = 0.0

        remaining = int(n_diag_done)
        nf = 0
        while remaining > 0:
            nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
            if not os.path.exists(nc_filename):
                print(f"[create_diag_nc] missing previous diagnostic file {nc_filename}; "
                      "running lam averages restart from available records")
                break
            with nc.Dataset(nc_filename, 'r', format='NETCDF4') as ds:
                if hasattr(ds, 'epsilon'):
                    self._adopt_diag_epsilon(ds, nc_filename)
                n_take = min(remaining, nsave, ds.variables['lam_i'].shape[0])
                if n_take <= 0:
                    break
                prev = np.asarray(ds.variables['lam_i'][:n_take], dtype=np.float64)
                prev_z = np.asarray(ds.variables['lam_i_z'][:n_take], dtype=np.float64)
                if 'prod_i' in ds.variables and 'diss_i' in ds.variables:
                    prev_prod = np.asarray(ds.variables['prod_i'][:n_take], dtype=np.float64)
                    prev_diss = np.asarray(ds.variables['diss_i'][:n_take], dtype=np.float64)
                elif 'tprodk' in ds.variables and 'tdissk' in ds.variables:
                    prev_prod = 0.5 * np.nansum(ds.variables['tprodk'][:n_take], axis=1)
                    prev_diss = -0.5 * np.nansum(ds.variables['tdissk'][:n_take], axis=1)
                else:
                    prev_prod = np.full(n_take, np.nan)
                    prev_diss = np.full(n_take, np.nan)
            self._lam_sum += float(np.nansum(prev))
            self._lam_n += int(np.isfinite(prev).sum())
            self._lamz_sum += float(np.nansum(prev_z))
            self._lamz_n += int(np.isfinite(prev_z).sum())
            self._prod_sum += float(np.nansum(prev_prod))
            self._diss_sum += float(np.nansum(prev_diss))
            self._lbudget_sum += float(np.nansum(prev_prod - prev_diss))
            remaining -= n_take
            nf += 1

    def save_diag(self, it, t_abs, lam_i, enorm, lam_i_z, znorm, budget):
        (evk, zvk, pvk, teadvk, teprodk, tefrick, tevisck,
         tzadvk, tzprodk, tzfrick, tzvisck,
         tpadvk, tpprodk, tpvisck, prod_i, diss_i) = budget
        tprodk = teadvk + teprodk
        tdissk = tefrick + tevisck
        lam_budget_i = prod_i - diss_i
        self.d_times[it] = t_abs
        self.lami_var[it] = lam_i
        self._lam_sum += lam_i
        self._lam_n += 1
        running_lam = self._lam_sum / self._lam_n
        self.lam_var[it] = running_lam
        self.enorm_var[it] = enorm
        self.lamiz_var[it] = lam_i_z
        self._lamz_sum += lam_i_z
        self._lamz_n += 1
        self.lamz_var[it] = self._lamz_sum / self._lamz_n
        self.znorm_var[it] = znorm
        self.evk_var[it, :] = evk
        self.zvk_var[it, :] = zvk
        self.pvk_var[it, :] = pvk
        self.teadvk_var[it, :] = teadvk
        self.teprodk_var[it, :] = teprodk
        self.tefrick_var[it, :] = tefrick
        self.tevisck_var[it, :] = tevisck
        self.tprodk_var[it, :] = tprodk
        self.tdissk_var[it, :] = tdissk
        self.tzadvk_var[it, :] = tzadvk
        self.tzprodk_var[it, :] = tzprodk
        self.tzfrick_var[it, :] = tzfrick
        self.tzvisck_var[it, :] = tzvisck
        self.tpadvk_var[it, :] = tpadvk
        self.tpprodk_var[it, :] = tpprodk
        self.tpvisck_var[it, :] = tpvisck
        self.prodi_var[it] = prod_i
        self.dissi_var[it] = diss_i
        self.lbudgeti_var[it] = lam_budget_i
        self.lbudget_residi_var[it] = lam_i - lam_budget_i
        self._prod_sum = getattr(self, '_prod_sum', 0.0) + prod_i
        self._diss_sum = getattr(self, '_diss_sum', 0.0) + diss_i
        self._lbudget_sum = getattr(self, '_lbudget_sum', 0.0) + lam_budget_i
        self.prod_var[it] = self._prod_sum / self._lam_n
        self.diss_var[it] = self._diss_sum / self._lam_n
        running_lbudget = self._lbudget_sum / self._lam_n
        self.lbudget_var[it] = running_lbudget
        self.lbudget_resid_var[it] = running_lam - running_lbudget
        self.dds.sync()

    def _bind_snapshot_var(self, name, description):
        if name in self.ds.variables:
            var = self.ds.variables[name]
        else:
            var = self.ds.createVariable(name, 'f4', ('time', 'y', 'x'), zlib=False)
        var.description = description
        return var

    def create_nc(self, nf, prefix='cle_o'):
        """Snapshot file with master q, raw perturbations, and normalized LLV."""
        nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
        if os.path.exists(nc_filename) and not self.is_not_rst:
            self.ds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            self.times = self.ds.variables['time']
            self.q_var = self.ds.variables['q']
            self.llv_var = self.ds.variables['llv']
            self.delta_psi_var = self._bind_snapshot_var('delta_psi', 'raw delta_psi')
            self.delta_q_var = self._bind_snapshot_var('delta_q', 'raw delta_q')
        else:
            self.ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            self.ds.createDimension('time', None)
            self.ds.createDimension('x', self.m_ref.Nx)
            self.ds.createDimension('y', self.m_ref.Ny)
            self.times = self.ds.createVariable('time', 'f4', ('time',))
            xs = self.ds.createVariable('x', 'f4', ('x',))
            ys = self.ds.createVariable('y', 'f4', ('y',))
            xs[:] = self.m_ref.x.get()
            ys[:] = self.m_ref.y.get()
            self.q_var = self.ds.createVariable('q', 'f4', ('time', 'y', 'x'), zlib=False)
            self.llv_var = self.ds.createVariable('llv', 'f4', ('time', 'y', 'x'), zlib=False)
            self.delta_psi_var = self._bind_snapshot_var('delta_psi', 'raw delta_psi')
            self.delta_q_var = self._bind_snapshot_var('delta_q', 'raw delta_q')
            self.ds.description = "QG CLE run snapshots"
            self.ds.Nobs = self.Nobs
            self.ds.dT_cle = self.dT
            self.ds.epsilon = self.epsilon
            self.ds.rescale_norm = "energy"
            self.ds.llv_description = "delta_psi normalized by the energy norm"
        self._stamp_output_metadata(self.ds)

    def save_var(self, it):
        self.times[it] = self.m_ref.t
        self.q_var[it, :, :] = ifft2(self.m_ref.q_hat).real.get()
        dq = self._delta_q()
        dpsi = self._delta_psi(dq)
        enorm = self._enorm(dpsi)
        dpsi_r = ifft2(dpsi).real
        dq_r = ifft2(dq).real
        self.delta_psi_var[it, :, :] = dpsi_r.get()
        self.delta_q_var[it, :, :] = dq_r.get()
        self.llv_var[it, :, :] = (dpsi_r / max(enorm, 1e-30)).get()
        self.ds.sync()

    def create_rst(self, nf):
        if self.is_not_rst:
            for prefix in ("rst_ref", "rst_cle"):
                path = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
                if os.path.exists(path):
                    os.remove(path)
        self.m_ref.create_rst(nf, prefix='rst_ref')
        self.m_cle.create_rst(nf, prefix='rst_cle')
        self._stamp_rst_metadata(self.m_ref.rstds, 'ref')
        self._stamp_rst_metadata(self.m_cle.rstds, 'cle')

    def _stamp_rst_metadata(self, ds, role):
        ds.cle_role = role
        ds.cle_precision = self.m_ref.precision
        ds.cle_Nobs = int(self.Nobs)
        ds.cle_dTobs = float(self.dTobs)
        ds.cle_dT_cle = float(self.dT)
        ds.cle_epsilon = float(self.epsilon)
        ds.cle_rescale_norm = "energy"
        ds.cle_dt = float(self.dt)
        ds.cle_Nx = int(self.m_ref.Nx)
        ds.cle_Ny = int(self.m_ref.Ny)
        ds.cle_seed = int(self.seed)
        ds.sync()

    def save_rst(self, it):
        self.m_ref.save_rst(it)
        self.m_cle.save_rst(it)

    def close_nc(self):
        self.ds.close()

    def close_diag(self):
        if getattr(self, 'dds', None) is not None:
            self.dds.close()
            self.dds = None

    def close_rst(self):
        self.m_ref.rstds.close()
        self.m_cle.rstds.close()

    def _set_model_times(self, rt):
        """Set absolute model time = pickup time from spinup + CLE relative time."""
        self.m_ref.t = self.m_ref.trst + rt
        self.m_cle.t = self.m_cle.trst + rt

    # ---------------- main loop ----------------
    def cle_run(self, scheme='ab3', tmax=40, tsave=200, tsave_rst=2000,
                nsave=100, savedir='run_cle0'):
        """Args mirror QGCDA.cda_run."""
        self.m_ref.ts_scheme = scheme
        self.m_cle.ts_scheme = scheme
        self.m_ref.savedir = savedir
        self.m_cle.savedir = savedir
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)

        if tsave_rst % self.intvl != 0:
            raise ValueError("tsave_rst must be a multiple of the rescaling interval "
                             f"({self.intvl} steps) so restarts align with rescales.")

        total_steps = int(round(tmax / self.dt))
        nrst = nsave
        ndiag = max(1, int(round((tsave * nsave) / self.intvl)))
        print(f"Starting CLE run. Nobs={self.Nobs}, dTobs={self.dTobs}, "
              f"dT_cle={self.dT}, epsilon={self.epsilon:.6e}, "
              f"rescale_norm=energy, tmax={tmax}")

        if self.is_not_rst:
            nf0 = nfrst0 = nfdiag0 = itsave = itrst = itdiag = 0
            insave = nsave
            inrst = nrst
            indiag = ndiag
            n_start = 0
            nf = nf0
            nfrst = nfrst0
            nfdiag = nfdiag0
            open_diag_midfile = False
        else:
            n_start_idx = int(round(self.rtrst / self.dt))
            hist_saves = n_start_idx // tsave
            nf0 = hist_saves // nsave
            itsave = hist_saves % nsave
            hist_saves_rst = n_start_idx // tsave_rst
            nfrst0 = hist_saves_rst // nrst
            itrst = hist_saves_rst % nrst
            hist_diag = n_start_idx // self.intvl
            nfdiag0 = hist_diag // ndiag
            itdiag = hist_diag % ndiag
            nf = nf0
            nfrst = nfrst0
            nfdiag = nfdiag0
            if itsave == 0:
                insave = nsave
            else:
                insave = itsave
                self.create_nc(nf0)
                nf += 1
            if itrst == 0:
                inrst = nrst
            else:
                inrst = itrst
                self.create_rst(nfrst0)
                nfrst += 1
            if itdiag == 0:
                indiag = ndiag
                open_diag_midfile = False
            else:
                indiag = itdiag
                open_diag_midfile = True
            n_start = n_start_idx

        # running average of lam_i; on resume, rebuild from existing records
        n_diag_done = n_start // self.intvl
        self._init_lam_sums(n_diag_done, ndiag)
        if open_diag_midfile:
            self.create_diag_nc(nfdiag0)
            nfdiag += 1
        # Norms of the current rescaled error; references for the next interval.
        dq0 = self._delta_q()
        dpsi0 = self._delta_psi(dq0)
        self._enorm_prev = self._enorm(dpsi0)
        self._znorm_prev = self._znorm(dq0)

        # Same clock convention as cda_turb2d.py: rt is CLE-relative runtime;
        # model.t is absolute time from the spinup pickup point.
        self.rt = n_start * self.dt
        self._set_model_times(self.rt)

        for n in range(n_start, total_steps + 1):
            self.rt = n * self.dt
            self.m_ref.n_steps = n
            self.m_cle.n_steps = n

            # exact insertion of the observed modes before anything else
            if n % self.intvl_da == 0:
                # p_hat/rv_hat are rebuilt by the immediately following model
                # step; interval diagnostics use q_hat directly.
                self._insert(sync_state=False)

            if n % 10000 == 0:
                E_r = self.m_ref.get_Etot(self.m_ref.p_hat) / self.m_ref.Nx / self.m_ref.Ny
                lam_e = self._lam_sum / self._lam_n if self._lam_n else float('nan')
                lam_z = self._lamz_sum / self._lamz_n if self._lamz_n else float('nan')
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}      "
                      f"step {n:7d}      t={self.m_ref.t:9.6f}s      E_ref={E_r:.4e}      "
                      f"lam={lam_e:.4e}      lam_z={lam_z:.4e}")

            # measure and rescale at the end of each dT_cle interval
            if n > n_start and n % self.intvl == 0:
                dq = self._delta_q()
                dpsi = self._delta_psi(dq)
                znorm = self._znorm(dq)
                enorm = self._enorm(dpsi)
                lam_i = np.log(enorm / max(self._enorm_prev, 1e-300)) / self.dT
                lam_i_z = np.log(znorm / max(self._znorm_prev, 1e-300)) / self.dT
                budget = self._err_budget(dpsi, enorm)
                if indiag == ndiag:
                    itdiag = 0
                    if nfdiag > nfdiag0:
                        self.close_diag()
                    self.create_diag_nc(nfdiag)
                    indiag = 0
                    nfdiag += 1
                self.save_diag(itdiag, float(self.m_ref.t), lam_i, enorm, lam_i_z, znorm, budget)
                itdiag += 1
                indiag += 1
                fac = self.epsilon / max(enorm, 1e-300)
                self._rescale(fac)
                # Rescaling is uniform, so both norms scale by the same factor.
                self._enorm_prev = enorm * fac
                self._znorm_prev = znorm * fac

            if n % tsave == 0:
                if insave == nsave:
                    itsave = 0
                    if nf > nf0:
                        self.close_nc()
                    self.create_nc(nf)
                    insave = 0
                    nf += 1
                self.save_var(itsave)
                print(f"[save_var] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   "
                      f"step {n:7d}  t={self.m_ref.t:9.6f}s ")
                itsave += 1
                insave += 1

            if n % tsave_rst == 0:
                if inrst == nrst:
                    itrst = 0
                    if nfrst > nfrst0:
                        self.close_rst()
                    self.create_rst(nfrst)
                    inrst = 0
                    nfrst += 1
                self.save_rst(itrst)
                print(f"[save_rst] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   "
                      f"step {n:7d}  t={self.m_ref.t:9.6f}s ")
                itrst += 1
                inrst += 1

            self.m_ref._step_forward()
            self.m_cle._step_forward()
            self.m_ref.t = self.m_ref.trst + (n + 1) * self.dt
            self.m_cle.t = self.m_cle.trst + (n + 1) * self.dt

        self.close_nc()
        self.close_rst()
        self.close_diag()
        print('Done.')
