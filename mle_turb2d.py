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


class QGMLE:
    """Maximal Lyapunov exponent and leading Lyapunov vector for QGModel.

    Two-trajectory (base + perturbed) method with periodic rescaling of the
    error back to a fixed small energy norm every dT_mle (Wolf et al. 1985;
    Boffetta & Musacchio 2017; Li et al. 2024 JFM 983 A1). The base model from
    turb2d.py is reused unmodified; the perturbed replica is a deepcopy whose
    streamfunction receives a random perturbation on all nonzero modes,
    converted back to q_hat for time stepping.

    Every dT_mle the module records the local exponents (Benettin/Inubushi lambda_i)
        lam_i   = ln(||delta||_E / ||delta||_E,prev) / dT_mle
        lam_i_z = ln(||delta||_Z / ||delta||_Z,prev) / dT_mle
    (||.||_E the energy norm of get_Etot, the fixed rescaling norm matching the
    velocity 2-norm used by Li/Inubushi; ||.||_Z the enstrophy norm of
    get_Ztot, i.e. the L2 norm of the state q), the running averages lam/lam_z, the
    normalized error (LLV) energy/enstrophy spectra, and the error energy and
    enstrophy budget spectra (advective transfer, production against the base
    flow, friction and hyperviscosity dissipation) as in Li et al. (2025 JFM
    1010 A12). The diagnostics also save Li-style combined production
    tprodk = teadvk + teprodk and scalar running averages of production,
    dissipation and prod - diss for comparison with lam. Both velocity-based
    quantities (evk, te*) and q-based
    diagnostic quantities (zvk, tz*) are evaluated from the same
    energy-normalized LLV; the q-side diagnostics are not separately normalized
    to unit enstrophy. Budget spectra are instantaneous endpoint diagnostics
    at the save time; they are not interval averages of lam_i.

    Snapshots save q of the base trajectory, raw delta_psi and delta_q, and
    the energy-normalized streamfunction error field
    llv = delta_psi/||delta||_E; no flux diagnostics are saved.
    """

    def __init__(self, m=None, m_pert=None, dT_mle=0.1,
                 epsilon_rel=1e-6, epsilon_abs=None, seed=10,
                 is_not_rst=True, rtrst=0.0):
        """Args:
            m: Base QGModel with initial condition already set.
            m_pert: Optional perturbed replica (pass when resuming from
                restart files). If None, it is created as deepcopy(m) plus a
                random all-nonzero-mode streamfunction perturbation of energy
                norm epsilon.
            dT_mle: Rescaling interval; also the diagnostic save cadence.
            epsilon_rel: Perturbation energy norm relative to the base state.
            epsilon_abs: Absolute perturbation energy norm (overrides rel).
            seed: CuPy RNG seed for the initial perturbation.
            is_not_rst: False to resume a previous run at time rtrst.
            rtrst: Relative resume time (multiple of dT_mle and tsave_rst*dt).
        """
        self.m = m
        self.dt = self.m.dt
        self.dT = float(dT_mle)
        self.intvl = int(round(self.dT / self.dt))
        self.is_not_rst = is_not_rst
        self.rtrst = rtrst
        self.seed = seed

        # shell-sum bookkeeping reused from the base model
        self._kk_idx_cpu = m.kk_idx.ravel().get()
        self._kk_sel = m.kk_range.get()
        # Parseval with the unnormalized FFT needs 1/(Nx*Ny)^2 so norms and
        # spectra are spatial averages (resolution-independent, comparable
        # across N and with the literature)
        self._norm_fac = 1.0 / (m.Nx * m.Ny) ** 2

        base_norm = self._enorm(m.p_hat)
        if epsilon_abs is not None:
            self.epsilon = float(epsilon_abs)
        else:
            self.epsilon = float(epsilon_rel) * max(base_norm, 1e-30)
        if self.epsilon <= 0.0:
            raise ValueError(f"Perturbation norm must be positive, got {self.epsilon}.")

        if m_pert is not None:
            self.m_pert = m_pert
        else:
            self.m_pert = copy.deepcopy(m)
            cp.random.seed(seed)
            self.m_pert.q_hat = self.m_pert.q_hat + self._make_random_delta(self._all_mode_mask())
            self._sync_state(self.m_pert)

    # ---------------- perturbation helpers ----------------
    def _all_mode_mask(self):
        mask = cp.ones((self.m.Ny, self.m.Nx), dtype=cp.float32)
        mask[0, 0] = 0.0
        return mask

    def _make_random_delta(self, mask):
        noise = cp.random.randn(self.m.Ny, self.m.Nx).astype(cp.float32)
        delta_psi_hat = fft2(noise).astype(cp.complex64)
        delta_psi_hat *= mask
        delta_psi_hat[0, 0] = 0.0
        norm = self._enorm(delta_psi_hat)
        if norm <= 0.0:
            raise RuntimeError("Random perturbation has zero energy norm after masking.")
        return self._psi_to_q((self.epsilon / norm) * delta_psi_hat)

    def _sync_state(self, model):
        model.p_hat = model.inversion * model.q_hat
        model.rv_hat = model.q_hat + model.gamma**2 * model.p_hat

    # ---------------- norms and spectra ----------------
    def _psi_to_q(self, dpsi_hat):
        return (self.m.lap - self.m.gamma**2) * dpsi_hat

    def _q_to_psi(self, dq_hat):
        return self.m.inversion * dq_hat

    def _enorm(self, dpsi_hat):
        """Energy norm of a streamfunction perturbation."""
        ene_dens = 0.5 * (self.m.kk**2 + self.m.gamma**2) * cp.abs(dpsi_hat)**2
        ek = self._shell_sum(ene_dens)
        return float(np.sqrt(max(np.sum(ek), 0.0)))

    def _znorm(self, dq_hat):
        """Enstrophy norm consistent with QGModel.get_Ztot (rounded shells)."""
        ens_dens = 0.5 * cp.abs(dq_hat)**2
        zk = self._shell_sum(ens_dens)
        return float(np.sqrt(max(np.sum(zk), 0.0)))

    def _shell_sum(self, dens):
        return npg.aggregate(self._kk_idx_cpu, dens.ravel().get(),
                             func='sum')[self._kk_sel] * self._norm_fac

    def _delta_q(self):
        return self.m_pert.q_hat - self.m.q_hat

    def _delta_psi(self, dq_hat=None):
        if dq_hat is None:
            dq_hat = self._delta_q()
        return self._q_to_psi(dq_hat)

    def _err_budget(self, dpsi_hat, enorm):
        """Spectra of the normalized error field (LLV) and its energy and
        enstrophy budget terms following Li et al. (2025b) eq. (2.27)-(2.29),
        adapted to the QG PV equation.

        Sign conventions follow get_TENL/get_diagFric/get_diagVisc of
        turb2d.py: each te*/tz* is the shell-integrated energy/enstrophy
        tendency of the normalized error from that process.
        """
        m = self.m
        # Energy-normalized LLV in streamfunction form; q/rv are derived from it
        # only because the QG equation is written for PV.
        epsi = dpsi_hat / max(enorm, 1e-300)
        eq = self._psi_to_q(epsi)
        erv = eq + m.gamma**2 * epsi

        # normalized error spectra (energy spectrum = LLV spectrum)
        evk = self._shell_sum(0.5 * (m.kk**2 + m.gamma**2) * cp.abs(epsi)**2)
        zvk = self._shell_sum(0.5 * cp.abs(eq)**2)

        # linearized nonlinear interaction with the base flow:
        # advective transfer J(psi_b, dq) and production J(dpsi, q_b)
        j_adv = m._compute_jacobian(m.p_hat, eq)
        j_prod = m._compute_jacobian(epsi, m.q_hat)
        teadvk = self._shell_sum(cp.real(cp.conj(epsi) * j_adv))
        teprodk = self._shell_sum(cp.real(cp.conj(epsi) * j_prod))
        tzadvk = self._shell_sum(-cp.real(cp.conj(eq) * j_adv))
        tzprodk = self._shell_sum(-cp.real(cp.conj(eq) * j_prod))

        # linear dissipation acting on the error (beta term is energy-neutral;
        # forcing is identical in both runs and cancels; Leith (cl!=0) and the
        # Arbic filter are not included in this budget)
        fric_term = -m.friction_mask * m.friction * erv
        visc_term = m.hylap * erv
        tefrick = self._shell_sum(-cp.real(cp.conj(epsi) * fric_term))
        tzfrick = self._shell_sum(cp.real(cp.conj(eq) * fric_term))
        tevisck = self._shell_sum(-cp.real(cp.conj(epsi) * visc_term))
        tzvisck = self._shell_sum(cp.real(cp.conj(eq) * visc_term))

        return evk, zvk, teadvk, teprodk, tefrick, tevisck, tzadvk, tzprodk, tzfrick, tzvisck

    def _rescale(self, fac):
        """Pull the perturbed trajectory back to distance epsilon from the
        base. The AB3 RHS history difference is rescaled by the same factor so
        the multistep tangent dynamics stays consistent across the rescale."""
        self.m_pert.q_hat = self.m.q_hat + fac * (self.m_pert.q_hat - self.m.q_hat)
        self.m_pert.k1_p = self.m.k1_p + fac * (self.m_pert.k1_p - self.m.k1_p)
        self.m_pert.k1_pp = self.m.k1_pp + fac * (self.m_pert.k1_pp - self.m.k1_pp)
        self._sync_state(self.m_pert)

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
        self.teadvk_var = self.dds.variables['teadvk']
        self.teprodk_var = self.dds.variables['teprodk']
        self.tefrick_var = self.dds.variables['tefrick']
        self.tevisck_var = self.dds.variables['tevisck']
        self.tzadvk_var = self.dds.variables['tzadvk']
        self.tzprodk_var = self.dds.variables['tzprodk']
        self.tzfrick_var = self.dds.variables['tzfrick']
        self.tzvisck_var = self.dds.variables['tzvisck']
        self.tprodk_var = self._bind_diag_var(
            'tprodk', 'f4', ('time', 'k'),
            'Li-style combined production spectrum teadvk + teprodk')
        self.tdissk_var = self._bind_diag_var(
            'tdissk', 'f4', ('time', 'k'),
            'combined energy dissipation tendency tefrick + tevisck')
        self.prodi_var = self._bind_diag_var(
            'prod_i', 'f8', ('time',),
            'instantaneous 0.5*sum_k(tprodk), comparable to Li production')
        self.dissi_var = self._bind_diag_var(
            'diss_i', 'f8', ('time',),
            'instantaneous -0.5*sum_k(tdissk), positive Li-style dissipation')
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

    def _adopt_diag_epsilon(self, ds, nc_filename):
        stored_eps = float(ds.epsilon)
        if not np.isclose(stored_eps, self.epsilon, rtol=1e-6):
            print(f"[create_diag_nc] adopting epsilon={stored_eps:.6e} from "
                  f"{nc_filename} (constructor gave {self.epsilon:.6e}) so the "
                  "rescaling target stays constant across restarts")
            self.epsilon = stored_eps

    def create_diag_nc(self, nf, prefix='mle_d'):
        nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
        if os.path.exists(nc_filename) and not self.is_not_rst:
            self.dds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            self._adopt_diag_epsilon(self.dds, nc_filename)
            self._bind_diag_vars()
        else:
            self.dds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            self.dds.createDimension('time', None)
            self.dds.createDimension('k', len(self.m.kk_iso))
            kk = self.dds.createVariable('k', 'f4', ('k',))
            kk[:] = self.m.kk_iso.get()
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
            self.dds.description = "QG MLE/LLV diagnostics (two-trajectory rescaling)"
            self.dds.dT_mle = self.dT
            self.dds.epsilon = self.epsilon
            self.dds.rescale_norm = "energy"
            self.dds.seed = self.seed
            self.dds.dt = self.dt
            self.dds.file_index = nf

    def _init_lam_sums(self, n_diag_done, nsave, prefix='mle_d'):
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
        (evk, zvk, teadvk, teprodk, tefrick, tevisck,
         tzadvk, tzprodk, tzfrick, tzvisck) = budget
        tprodk = teadvk + teprodk
        tdissk = tefrick + tevisck
        # Our LLV is normalized to energy 1, so the integrated normalized
        # tendency is 2*lambda. The factor 0.5 maps the scalar budget to Li's
        # convention where the normalized kinetic energy is 1/2.
        prod_i = 0.5 * float(np.nansum(tprodk))
        diss_i = -0.5 * float(np.nansum(tdissk))
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

    def create_nc(self, nf, prefix='mle_o'):
        """Snapshot file with base q, raw perturbations, and normalized LLV."""
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
            self.ds.createDimension('x', self.m.Nx)
            self.ds.createDimension('y', self.m.Ny)
            self.times = self.ds.createVariable('time', 'f4', ('time',))
            xs = self.ds.createVariable('x', 'f4', ('x',))
            ys = self.ds.createVariable('y', 'f4', ('y',))
            xs[:] = self.m.x.get()
            ys[:] = self.m.y.get()
            self.q_var = self.ds.createVariable('q', 'f4', ('time', 'y', 'x'), zlib=False)
            self.llv_var = self.ds.createVariable('llv', 'f4', ('time', 'y', 'x'), zlib=False)
            self.delta_psi_var = self._bind_snapshot_var('delta_psi', 'raw delta_psi')
            self.delta_q_var = self._bind_snapshot_var('delta_q', 'raw delta_q')
            self.ds.description = "QG MLE run snapshots"
            self.ds.dT_mle = self.dT
            self.ds.epsilon = self.epsilon
            self.ds.rescale_norm = "energy"
            self.ds.llv_description = "delta_psi normalized by the energy norm"

    def save_var(self, it):
        self.times[it] = self.m.t
        self.q_var[it, :, :] = ifft2(self.m.q_hat).real.get()
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
            for prefix in ("rst_base", "rst_pert"):
                path = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
                if os.path.exists(path):
                    os.remove(path)
        self.m.create_rst(nf, prefix='rst_base')
        self.m_pert.create_rst(nf, prefix='rst_pert')

    def save_rst(self, it):
        self.m.save_rst(it)
        self.m_pert.save_rst(it)

    def close_nc(self):
        self.ds.close()

    def close_diag(self):
        if getattr(self, 'dds', None) is not None:
            self.dds.close()
            self.dds = None

    def close_rst(self):
        self.m.rstds.close()
        self.m_pert.rstds.close()

    def _set_model_times(self, rt):
        """Set absolute model time = pickup time from spinup + LE relative time."""
        self.m.t = self.m.trst + rt
        self.m_pert.t = self.m_pert.trst + rt

    # ---------------- main loop ----------------
    def mle_run(self, scheme='ab3', tmax=40, tsave=200, tsave_rst=2000,
                nsave=100, savedir='run_mle0'):
        """Args mirror QGCDA.cda_run."""
        self.m.ts_scheme = scheme
        self.m_pert.ts_scheme = scheme
        self.m.savedir = savedir
        self.m_pert.savedir = savedir
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)

        if tsave_rst % self.intvl != 0:
            raise ValueError("tsave_rst must be a multiple of the rescaling interval "
                             f"({self.intvl} steps) so restarts align with rescales.")

        total_steps = int(round(tmax / self.dt))
        nrst = nsave
        ndiag = max(1, int(round((tsave * nsave) / self.intvl)))
        print(f"Starting MLE run. dT_mle={self.dT}, epsilon={self.epsilon:.6e}, "
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

        # Same clock convention as cda_turb2d.py: rt is LE-relative runtime;
        # model.t is absolute time from the spinup pickup point.
        self.rt = n_start * self.dt
        self._set_model_times(self.rt)

        for n in range(n_start, total_steps + 1):
            self.rt = n * self.dt
            self.m.n_steps = n
            self.m_pert.n_steps = n

            if n % 10000 == 0:
                E_b = self.m.get_Etot(self.m.p_hat) / self.m.Nx / self.m.Ny
                lam_e = self._lam_sum / self._lam_n if self._lam_n else float('nan')
                lam_z = self._lamz_sum / self._lamz_n if self._lamz_n else float('nan')
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}      "
                      f"step {n:7d}      t={self.m.t:9.6f}s      E_base={E_b:.4e}      "
                      f"lam={lam_e:.4e}      lam_z={lam_z:.4e}")

            # measure and rescale at the end of each dT_mle interval
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
                self.save_diag(itdiag, float(self.m.t), lam_i, enorm, lam_i_z, znorm, budget)
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
                      f"step {n:7d}  t={self.m.t:9.6f}s ")
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
                      f"step {n:7d}  t={self.m.t:9.6f}s ")
                itrst += 1
                inrst += 1

            self.m._step_forward()
            self.m_pert._step_forward()
            self.m.t = self.m.trst + (n + 1) * self.dt
            self.m_pert.t = self.m_pert.trst + (n + 1) * self.dt

        self.close_nc()
        self.close_rst()
        self.close_diag()
        print('Done.')
