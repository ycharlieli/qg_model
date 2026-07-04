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
    insertion: at every dTobs the slave's observed modes (rounded shells
    k <= Nobs) are replaced by the master's, so the error lives entirely in
    the unobserved subspace (Li et al. 2025b eq. 2.5; Inubushi et al. SM,
    delta_p == 0). The slave starts epsilon-close to the master and the error
    is rescaled back to epsilon every dT_cle, the renormalized two-trajectory
    method of Boffetta & Musacchio (2017) / Li et al. (2024), so both signs of
    the CLE are measurable indefinitely.

    Every dT_cle the module records the local exponents (Benettin/Inubushi lambda_i)
        lam_i_z = ln(||delta||_Z / epsilon) / dT_cle
        lam_i   = ln(||delta||_E / ||delta||_E,prev) / dT_cle
    (||.||_Z the enstrophy norm, i.e. the L2 norm of the state q, which the
    rescaling holds at epsilon as in Inubushi et al. SM eq. S8; ||.||_E the
    energy norm), the running averages lam_z/lam (both estimate the CLE), the
    error norms znorm and enorm (synchronisation error norms of the
    literature, not RMSE), the normalized error (conditional LLV) energy/enstrophy
    spectra, and the error budget spectra (advective transfer, production,
    friction and hyperviscosity dissipation) as in Li et al. (2025b).
    Velocity-based quantities (evk, te*) are normalized by the energy norm;
    q-based quantities (zvk, tz*, llv) by the enstrophy norm, since q is a
    vorticity. Consistency checks:
        lam_i   ~= 0.5*sum_k(teadvk+teprodk+tefrick+tevisck)
        lam_i_z ~= 0.5*sum_k(tzadvk+tzprodk+tzfrick+tzvisck)

    Snapshots save only q of the master and the normalized error field
    llv = delta_q/||delta_q||_Z; no flux diagnostics, Ihm/Ihref or daF are
    saved.
    """

    def __init__(self, m_ref=None, m_cle=None, Nobs=16, dTobs=None, dT_cle=0.1,
                 epsilon_rel=1e-4, epsilon_abs=None, seed=10,
                 is_not_rst=True, rtrst=0.0):
        """Args:
            m_ref: Master (truth) QGModel with initial condition already set.
            m_cle: Optional slave (pass when resuming from restart files). If
                None, it is created as deepcopy(m_ref) plus a random
                perturbation confined to shells k > Nobs with enstrophy norm
                epsilon.
            Nobs: Observation cutoff wavenumber (rounded shells k <= Nobs are
                inserted from the master).
            dTobs: Insertion interval; defaults to the model time step.
            dT_cle: Rescaling interval; also the diagnostic save cadence.
            epsilon_rel: Perturbation enstrophy norm relative to the master state.
            epsilon_abs: Absolute perturbation enstrophy norm (overrides rel).
            seed: CuPy RNG seed for the initial perturbation.
            is_not_rst: False to resume a previous run at time rtrst.
            rtrst: Relative resume time (multiple of dT_cle and tsave_rst*dt).
        """
        self.m_ref = m_ref
        self.dt = float(m_ref.dt)
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

        # observed / unobserved spectral masks with the same rounded shells as
        # the diagnostics and cda_turb2d
        k_shell = cp.round(cp.sqrt(m_ref.nx2d**2 + m_ref.ny2d**2))
        self.obs_mask = (k_shell <= float(Nobs)).astype(cp.float32)
        self.unobs_mask = 1.0 - self.obs_mask
        self.unobs_mask[0, 0] = 0.0

        # shell-sum bookkeeping reused from the master model
        self._kk_idx_cpu = m_ref.kk_idx.ravel().get()
        self._kk_sel = m_ref.kk_range.get()
        self._norm_fac = 1.0 / (m_ref.Nx * m_ref.Ny)

        base_norm = self._znorm(m_ref.q_hat)
        if epsilon_abs is not None:
            self.epsilon = float(epsilon_abs)
        else:
            self.epsilon = float(epsilon_rel) * max(base_norm, 1e-30)
        if self.epsilon <= 0.0:
            raise ValueError(f"Perturbation norm must be positive, got {self.epsilon}.")

        if m_cle is not None:
            self.m_cle = m_cle
        else:
            self.m_cle = copy.deepcopy(m_ref)
            cp.random.seed(seed)
            self.m_cle.q_hat = self.m_cle.q_hat + self._make_random_delta(self.unobs_mask)
            self._sync_state(self.m_cle)

    # ---------------- perturbation and coupling helpers ----------------
    def _make_random_delta(self, mask):
        noise = cp.random.randn(self.m_ref.Ny, self.m_ref.Nx).astype(cp.float32)
        delta_hat = fft2(noise).astype(cp.complex64)
        delta_hat *= mask
        delta_hat[0, 0] = 0.0
        norm = self._znorm(delta_hat)
        if norm <= 0.0:
            raise RuntimeError("Random perturbation has zero enstrophy after masking.")
        return (self.epsilon / norm) * delta_hat

    def _sync_state(self, model):
        model.p_hat = model.inversion * model.q_hat
        model.rv_hat = model.q_hat + model.gamma**2 * model.p_hat

    def _insert(self):
        """OT2003 exact insertion of the observed modes of the master."""
        self.m_cle.q_hat = (self.m_cle.q_hat * self.unobs_mask
                            + self.m_ref.q_hat * self.obs_mask)
        self._sync_state(self.m_cle)

    # ---------------- norms and spectra ----------------
    def _enorm(self, dq_hat):
        """Energy norm consistent with QGModel.get_Etot (rounded shells)."""
        dp_hat = self.m_ref.inversion * dq_hat
        ene_dens = 0.5 * (self.m_ref.kk**2 + self.m_ref.gamma**2) * cp.abs(dp_hat)**2
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
        return (self.m_cle.q_hat - self.m_ref.q_hat) * self.unobs_mask

    def _err_budget(self, dq_hat, enorm, znorm):
        """Spectra of the normalized error (conditional LLV) and its energy
        and enstrophy budget terms, linearized about the master trajectory
        (Li et al. 2025b eq. 2.27-2.29 adapted to the QG PV equation). Sign
        conventions follow get_TENL/get_diagFric/get_diagVisc of turb2d.py."""
        m = self.m_ref
        # unit-enstrophy error: dq normalized by its own L2 (enstrophy) norm,
        # since q is a vorticity
        vq = dq_hat / max(znorm, 1e-300)
        vp = m.inversion * vq
        vrv = vq + m.gamma**2 * vp
        # quadratic E-side (velocity-based) quantities are re-normalized by
        # the energy norm
        fac_e = (znorm / max(enorm, 1e-300))**2

        # normalized error spectra (energy spectrum = conditional LLV spectrum)
        evk = self._shell_sum(0.5 * (m.kk**2 + m.gamma**2) * cp.abs(vp)**2) * fac_e
        zvk = self._shell_sum(0.5 * cp.abs(vq)**2)

        # advective transfer J(psi_ref, dq) and production J(dpsi, q_ref)
        j_adv = m._compute_jacobian(m.p_hat, vq)
        j_prod = m._compute_jacobian(vp, m.q_hat)
        teadvk = self._shell_sum(cp.real(cp.conj(vp) * j_adv)) * fac_e
        teprodk = self._shell_sum(cp.real(cp.conj(vp) * j_prod)) * fac_e
        tzadvk = self._shell_sum(-cp.real(cp.conj(vq) * j_adv))
        tzprodk = self._shell_sum(-cp.real(cp.conj(vq) * j_prod))

        # linear dissipation acting on the error (beta term is energy-neutral;
        # forcing is identical in both runs and cancels; Leith and the Arbic
        # filter are not included in this budget)
        fric_term = -m.friction_mask * m.friction * vrv
        visc_term = m.hylap * vrv
        tefrick = self._shell_sum(-cp.real(cp.conj(vp) * fric_term)) * fac_e
        tzfrick = self._shell_sum(cp.real(cp.conj(vq) * fric_term))
        tevisck = self._shell_sum(-cp.real(cp.conj(vp) * visc_term)) * fac_e
        tzvisck = self._shell_sum(cp.real(cp.conj(vq) * visc_term))

        return evk, zvk, teadvk, teprodk, tefrick, tevisck, tzadvk, tzprodk, tzfrick, tzvisck

    def _rescale(self, fac):
        """Pull the slave back to distance epsilon from the master. The AB3
        RHS history difference is rescaled by the same factor so the multistep
        tangent dynamics stays consistent across the rescale."""
        self.m_cle.q_hat = self.m_ref.q_hat + fac * (self.m_cle.q_hat - self.m_ref.q_hat)
        self.m_cle.k1_p = self.m_ref.k1_p + fac * (self.m_cle.k1_p - self.m_ref.k1_p)
        self.m_cle.k1_pp = self.m_ref.k1_pp + fac * (self.m_cle.k1_pp - self.m_ref.k1_pp)
        self._sync_state(self.m_cle)

    # ---------------- output ----------------
    def create_diag_nc(self, savedir, prefix='clediag'):
        nc_filename = os.path.join(savedir, f"{prefix}.nc")
        if os.path.exists(nc_filename):
            self.dds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            stored_eps = float(self.dds.epsilon)
            if not np.isclose(stored_eps, self.epsilon, rtol=1e-6):
                print(f"[create_diag_nc] adopting epsilon={stored_eps:.6e} from "
                      f"{nc_filename} (constructor gave {self.epsilon:.6e}) so the "
                      "rescaling target stays constant across restarts")
                self.epsilon = stored_eps
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
            self.dds.description = "QG CLE/conditional-LLV diagnostics (master-slave, rescaled)"
            self.dds.Nobs = self.Nobs
            self.dds.dTobs = self.dTobs
            self.dds.dT_cle = self.dT
            self.dds.epsilon = self.epsilon
            self.dds.control_norm = "enstrophy"
            self.dds.seed = self.seed
            self.dds.dt = self.dt

    def save_diag(self, it, t_abs, lam_i, enorm, lam_i_z, znorm, budget):
        (evk, zvk, teadvk, teprodk, tefrick, tevisck,
         tzadvk, tzprodk, tzfrick, tzvisck) = budget
        self.d_times[it] = t_abs
        self.lami_var[it] = lam_i
        self._lam_sum += lam_i
        self._lam_n += 1
        self.lam_var[it] = self._lam_sum / self._lam_n
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
        self.tzadvk_var[it, :] = tzadvk
        self.tzprodk_var[it, :] = tzprodk
        self.tzfrick_var[it, :] = tzfrick
        self.tzvisck_var[it, :] = tzvisck
        self.dds.sync()

    def create_nc(self, nf, prefix='cle_o'):
        """Snapshot file: only q of the master and the llv field."""
        nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
        if os.path.exists(nc_filename):
            self.ds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            self.times = self.ds.variables['time']
            self.q_var = self.ds.variables['q']
            self.llv_var = self.ds.variables['llv']
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
            self.ds.description = "QG CLE run snapshots"
            self.ds.Nobs = self.Nobs
            self.ds.dT_cle = self.dT
            self.ds.epsilon = self.epsilon

    def save_var(self, it):
        self.times[it] = self.m_ref.t
        self.q_var[it, :, :] = ifft2(self.m_ref.q_hat).real.get()
        dq = self._delta_q()
        znorm = self._znorm(dq)
        self.llv_var[it, :, :] = (ifft2(dq).real / max(znorm, 1e-30)).get()
        self.ds.sync()

    def create_rst(self, nf):
        self.m_ref.create_rst(nf, prefix='rst_ref')
        self.m_cle.create_rst(nf, prefix='rst_cle')

    def save_rst(self, it):
        self.m_ref.save_rst(it)
        self.m_cle.save_rst(it)

    def close_nc(self):
        self.ds.close()

    def close_rst(self):
        self.m_ref.rstds.close()
        self.m_cle.rstds.close()

    # ---------------- main loop ----------------
    def cle_run(self, scheme='ab3', tmax=40, tsave=200, tsave_rst=2000,
                nsave=100, savedir='run_cle0'):
        """Args mirror QGCDA.cda_run; tsave/tsave_rst in steps."""
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
        print(f"Starting CLE run. Nobs={self.Nobs}, dTobs={self.dTobs}, "
              f"dT_cle={self.dT}, epsilon={self.epsilon:.6e}, tmax={tmax}")

        if self.is_not_rst:
            nf0 = nfrst0 = itsave = itrst = 0
            insave = nsave
            inrst = nrst
            n_start = 0
        else:
            n_start = int(round(self.rtrst / self.dt))
            hist_saves = n_start // tsave
            nf0 = hist_saves // nsave
            itsave = hist_saves % nsave
            hist_saves_rst = n_start // tsave_rst
            nfrst0 = hist_saves_rst // nrst
            itrst = hist_saves_rst % nrst
            insave = nsave if itsave == 0 else itsave
            inrst = nrst if itrst == 0 else itrst
        nf = nf0
        nfrst = nfrst0
        if not self.is_not_rst:
            if itsave != 0:
                self.create_nc(nf0)
                nf += 1
            if itrst != 0:
                self.create_rst(nfrst0)
                nfrst += 1

        # running average of lam_i; on resume, rebuild from existing records
        self.create_diag_nc(savedir)
        n_diag_done = n_start // self.intvl
        if n_diag_done > 0 and self.d_times.shape[0] >= n_diag_done:
            prev = np.asarray(self.lami_var[:n_diag_done], dtype=np.float64)
            self._lam_sum = float(np.nansum(prev))
            self._lam_n = int(np.isfinite(prev).sum())
            prev_z = np.asarray(self.lamiz_var[:n_diag_done], dtype=np.float64)
            self._lamz_sum = float(np.nansum(prev_z))
            self._lamz_n = int(np.isfinite(prev_z).sum())
        else:
            self._lam_sum = 0.0
            self._lam_n = 0
            self._lamz_sum = 0.0
            self._lamz_n = 0
        # energy norm of the current (rescaled) error, reference point for lam_i
        self._enorm_prev = self._enorm(self._delta_q())

        for n in range(n_start, total_steps + 1):
            self.rt = n * self.dt
            self.m_ref.n_steps = n
            self.m_cle.n_steps = n
            self.m_ref.t = self.m_ref.trst + self.rt
            self.m_cle.t = self.m_cle.trst + self.rt

            # exact insertion of the observed modes before anything else
            if n % self.intvl_da == 0:
                self._insert()

            if n % 10000 == 0:
                E_r = self.m_ref.get_Etot(self.m_ref.p_hat) / self.m_ref.Nx / self.m_ref.Ny
                lam_e = self._lam_sum / self._lam_n if self._lam_n else float('nan')
                lam_z = self._lamz_sum / self._lamz_n if self._lamz_n else float('nan')
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}      "
                      f"step {n:7d}      t={self.m_ref.t:9.6f}s      E_ref={E_r:.4e}      "
                      f"lam_z={lam_z:.4e}      lam={lam_e:.4e}")

            # measure and rescale at the end of each dT_cle interval
            if n > n_start and n % self.intvl == 0:
                dq = self._delta_q()
                znorm = self._znorm(dq)
                lam_i_z = np.log(znorm / self.epsilon) / self.dT
                enorm = self._enorm(dq)
                lam_i = np.log(enorm / max(self._enorm_prev, 1e-300)) / self.dT
                budget = self._err_budget(dq, enorm, znorm)
                it_d = n // self.intvl - 1
                self.save_diag(it_d, float(self.m_ref.t), lam_i, enorm, lam_i_z, znorm, budget)
                self._rescale(self.epsilon / znorm)
                # rescaling is uniform, so the energy norm scales by the same factor
                self._enorm_prev = enorm * (self.epsilon / znorm)

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
        self.dds.close()
        print('Done.')
