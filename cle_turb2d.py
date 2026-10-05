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


BUDGET_REVISION = "generalized-velocity-l2-v2"
TRANSFER_NAMES = (
    "P_plus", "P_minus",
    "P_strain_source_receiver", "P_velocity_source_receiver",
)


class QGCLE:
    """Conditional Lyapunov exponent and conditional LLV for QGModel.

    Master-slave setup mimicking cda_turb2d.py with the OT2003 exact spectral
    insertion: at every dTobs the slave's observed modes (|k| < Nobs) are
    replaced by the master's, so the error lives entirely in
    the unobserved subspace (Li et al. 2025b, eq. 2.5). The slave starts
    epsilon-close to the master and the error
    is rescaled back to a fixed small velocity L2 norm every dT_cle, the renormalized
    two-trajectory
    method of Boffetta & Musacchio (2017) / Li et al. (2024), so both signs of
    the CLE are measurable indefinitely.

    Every dT_cle the module records finite-interval logarithmic growth rates
        lam_i   = ln(||delta||_2 / ||delta||_2,prev) / dT_cle
    (||.||_2^2 = <|v|^2>, the fixed rescaling norm), the running average
    lam, and the error norm l2norm (a synchronisation error norm of the
    literature, not RMSE), the normalized error (conditional LLV) energy,
    enstrophy and PV-gradient palinstrophy spectra, and their exact linearized
    spectral-budget terms following Li et al. (2025b), adapted to the QG PV
    equation. No cascade, principal-strain or alignment mechanism is assumed.
    The diagnostics save generalized-energy production
    tprodk = teadvk + teprodk, kinetic production tkproductionk, and scalar
    running averages of kinetic production, dissipation and prod - diss
    for comparison with lam. Both velocity-based
    quantities (kvk, tk*) and generalized-energy (evk, te*) and q-based
    diagnostic quantities (zvk, pvk, tz*,
    tp*) are evaluated from the same velocity-L2-normalized conditional LLV; the
    q-side diagnostics are not separately normalized to unit enstrophy or
    palinstrophy. Palinstrophy densities are weighted by each Fourier mode's
    exact k^2 before shell aggregation. For the target gamma=0, all-mode-drag
    runs, the omitted drag tendency is exactly -2*friction*pvk mode by mode;
    no redundant combined or friction palinstrophy arrays are stored.
    Budget spectra are instantaneous endpoint diagnostics at the save time;
    they are not interval averages of lam_i.

    Strain production P_plus/P_minus and the strain/actual-velocity
    source-receiver matrices are saved in cle_d at the same cadence as lam.
    Suffix _i denotes instantaneous endpoint values; unsuffixed variables are
    cumulative means over the same samples as lam, preserved across restarts.
    Net strain production is P_plus-P_minus, and receiver spectra follow by
    summing the matrices over source shells. No alignment statistics, duplicate
    spatial fields, or separate cle_svp files are saved.

    Snapshots save q of the master, raw delta_psi and delta_q, and the
    velocity-L2-normalized streamfunction error field
    llv = delta_psi/||delta||_2; no flux diagnostics, Ihm/Ihref or daF are
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
                velocity L2 norm epsilon.
            Nobs: Observation cutoff wavenumber (modes with |k| < Nobs are
                inserted from the master).
            dTobs: Insertion interval; defaults to the model time step.
            dT_cle: Rescaling interval; also the diagnostic save cadence.
            epsilon_rel: Perturbation velocity L2 norm relative to the master state.
            epsilon_abs: Absolute perturbation velocity L2 norm (overrides rel).
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

        base_norm = self._l2norm(m_ref.p_hat)
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
        self._reference_forcing = m_ref.forcing_norm in ('energy', 'enstrophy', 'kinetic_energy')
        if self._reference_forcing:
            for name in ('Nx', 'Ny', 'Lx', 'Ly', 'dt', 'trst'):
                if getattr(self.m_cle, name) != getattr(m_ref, name):
                    raise ValueError(f'Reference-normalized forcing requires matching {name}.')
            # Reuse the reference forcing instead of normalizing on the slave.
            self.m_cle.forcing = None
            self.m_cle.forcing_norm = None
            self.m_cle.force_q = m_ref.force_q.copy()

    # ---------------- perturbation and coupling helpers ----------------
    def _make_random_delta(self, mask):
        noise = cp.random.randn(self.m_ref.Ny, self.m_ref.Nx).astype(self.m_ref.rdtype)
        delta_psi_hat = fft2(noise).astype(self.m_ref.cdtype)
        delta_psi_hat *= mask
        delta_psi_hat[0, 0] = 0.0
        self.m_ref._enforce_spectral_constraints(delta_psi_hat)
        norm = self._l2norm(delta_psi_hat)
        if norm <= 0.0:
            raise RuntimeError("Random perturbation has zero velocity L2 norm after masking.")
        return self._psi_to_q((self.epsilon / norm) * delta_psi_hat)

    def _sync_state(self, model):
        model._enforce_spectral_constraints(model.q_hat)
        model.p_hat = model.inversion * model.q_hat
        model.rv_hat = model._relative_vorticity(model.p_hat, model.q_hat)

    def _insert(self, sync_state=True):
        """OT2003 exact insertion of the observed modes of the master."""
        cp.copyto(self.m_cle.q_hat, self.m_ref.q_hat, where=self.obs_bool)
        if sync_state:
            self._sync_state(self.m_cle)

    # ---------------- norms and spectra ----------------
    def _psi_to_q(self, dpsi_hat):
        return self.m_ref._q_from_psi(dpsi_hat)

    def _q_to_psi(self, dq_hat):
        return self.m_ref.inversion * dq_hat

    def _l2norm(self, dpsi_hat):
        """Velocity L2 norm, sqrt(<|v|^2>), of a streamfunction perturbation."""
        l2_dens = self.m_ref.kk**2 * cp.abs(dpsi_hat)**2
        l2_squared = self._norm_fac * float(
            cp.sum(l2_dens, dtype=cp.float64).get()
        )
        return float(np.sqrt(max(l2_squared, 0.0)))

    def _shell_sum(self, dens):
        return npg.aggregate(self._kk_idx_cpu, dens.ravel().get(),
                             func='sum')[self._kk_sel] * self._norm_fac

    def _delta_q(self):
        return (self.m_cle.q_hat - self.m_ref.q_hat) * self.unobs_mask

    def _delta_psi(self, dq_hat=None):
        if dq_hat is None:
            dq_hat = self._delta_q()
        return self._q_to_psi(dq_hat)

    def _err_budget(self, dpsi_hat, l2norm):
        """Spectra of the normalized error (conditional LLV) and its energy,
        enstrophy and PV-gradient palinstrophy budget terms, linearized about
        the master trajectory (Li et al. 2025b, eqs. 2.27--2.29, adapted to
        the QG PV equation). Sign conventions follow
        get_TENL/get_diagFric/get_diagVisc of turb2d.py."""
        m = self.m_ref
        # Velocity-L2-normalized LLV in streamfunction form; q/rv are derived from it
        # only because the QG equation is written for PV.
        epsi = dpsi_hat / max(l2norm, 1e-300)
        eq = self._psi_to_q(epsi)

        # All spectra use the same velocity-L2-normalized LLV.
        # Apply the exact modal k^2 before shell aggregation: rounded shell
        # centres are not exact substitutes for the modal wavenumbers.
        if m.alpha == 2:
            energy_density = 0.5 * (m.kk**2 + m.gamma**2) * cp.abs(epsi)**2
        else:
            energy_density = -0.5 * cp.real(cp.conj(epsi) * eq)
        evk = self._shell_sum(energy_density)
        kvk = self._shell_sum(0.5 * m.kk**2 * cp.abs(epsi)**2)
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
        fric_term = -m.friction_mask * m.friction * eq
        visc_term = m.hylap * eq
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

        # Kinetic work uses k^2/A(k), since K and E coincide only for
        # alpha=2, gamma=0. Keep that case's arithmetic unchanged.
        kinetic_weight = 1.0 if m.alpha == 2 and not m.gamma else -m.kk**2 * m.inversion
        tkadv, tkprod = kinetic_weight * teadv, kinetic_weight * teprod
        tkfric, tkvisc = kinetic_weight * tefric, kinetic_weight * tevisc
        tkadvk, tkprodk = self._shell_sum(tkadv), self._shell_sum(tkprod)
        tkfrick, tkvisck = self._shell_sum(tkfric), self._shell_sum(tkvisc)

        # Integrated budget terms include every Fourier mode; the saved *k
        # arrays remain isotropic spectra on complete radial shells.
        prod_i = self._norm_fac * float(
            cp.sum(tkadv + tkprod, dtype=cp.float64).get())
        diss_i = -self._norm_fac * float(
            cp.sum(tkfric + tkvisc, dtype=cp.float64).get())

        return (evk, zvk, pvk, teadvk, teprodk, tefrick, tevisck,
                tzadvk, tzprodk, tzfrick, tzvisck,
                tpadvk, tpprodk, tpvisck, prod_i, diss_i,
                kvk, tkadvk, tkprodk, tkfrick, tkvisck)

    # ---------------- runtime strain/transfer diagnostics ----------------
    def _transfer_shell_contract(self):
        """Complete radial shells plus one square-corner residual shell."""
        cached = getattr(self, "_transfer_shell_cache", None)
        if cached is not None:
            return cached
        m = self.m_ref
        if m.Nx != m.Ny or m.Nx % 2:
            raise ValueError("Runtime CLE strain diagnostics require an even square grid.")
        if not np.isclose(float(m.Lx), float(m.Ly), rtol=0.0, atol=1.0e-12):
            raise ValueError("Runtime CLE strain diagnostics require Lx=Ly.")
        n_complete = m.Nx // 2
        radial_index = cp.rint(
            cp.sqrt(m.nx2d.astype(cp.float64) ** 2
                    + m.ny2d.astype(cp.float64) ** 2)
        ).astype(cp.int32)
        shell_id = cp.minimum(radial_index, n_complete)
        n_shells = n_complete + 1
        dk = 2.0 * np.pi / float(m.Lx)
        kx_max = float(cp.max(cp.abs(m.kx2d)).get())
        ky_max = float(cp.max(cp.abs(m.ky2d)).get())
        corner = float(np.hypot(kx_max, ky_max))
        cached = {
            "shell_id": shell_id,
            "n_complete": n_complete,
            "n_shells": n_shells,
            "index": np.arange(n_shells, dtype=np.int32),
            "k": np.concatenate((
                np.arange(n_complete, dtype=np.float64) * dk,
                np.asarray([np.nan]),
            )),
            "edge": np.concatenate((
                np.asarray([0.0]),
                (np.arange(n_complete, dtype=np.float64) + 0.5) * dk,
                np.asarray([np.nextafter(corner, np.inf)]),
            )),
            "is_outer_residual": np.concatenate((
                np.zeros(n_complete, dtype=np.int8),
                np.ones(1, dtype=np.int8),
            )),
        }
        self._transfer_shell_cache = cached
        return cached

    def _padded_real(self, field_hat):
        """3/2-padded real fields, consuming the solver pad buffer immediately."""
        field_hat = cp.asarray(field_hat)
        if field_hat.ndim == 2:
            return ifft2(self.m_ref._padding(field_hat)).real
        flat = field_hat.reshape((-1, self.m_ref.Ny, self.m_ref.Nx))
        padded = [ifft2(self.m_ref._padding(component)).real for component in flat]
        return cp.stack(padded, axis=0).reshape(
            field_hat.shape[:-2] + (self.m_ref.Nypad, self.m_ref.Nxpad)
        )

    def _unpad_fft(self, padded_field):
        """FFT a padded real field and return its solver-grid spectrum."""
        padded_field = cp.asarray(padded_field)
        if padded_field.ndim == 2:
            return self.m_ref._unpadding(fft2(padded_field))
        flat = padded_field.reshape(
            (-1, self.m_ref.Nypad, self.m_ref.Nxpad)
        )
        spectra = [self.m_ref._unpadding(fft2(component)) for component in flat]
        return cp.stack(spectra, axis=0).reshape(
            padded_field.shape[:-2] + (self.m_ref.Ny, self.m_ref.Nx)
        )

    def _transfer_shell_sum(self, modal_values, shell):
        return cp.bincount(
            shell["shell_id"].ravel(), weights=modal_values.ravel(),
            minlength=shell["n_shells"],
        )[:shell["n_shells"]]

    def _compute_transfer_snapshot(self, dpsi_hat, l2norm):
        """Compute strain geometry and generalized velocity-error transfers."""
        m = self.m_ref
        shell = self._transfer_shell_contract()
        shell_id = shell["shell_id"]
        n_shells = shell["n_shells"]

        psi_hat = m.inversion * m.q_hat
        llv_hat = dpsi_hat / max(l2norm, 1.0e-300)
        llv_q_hat = self._psi_to_q(llv_hat)
        u_hat = cp.stack((
            -1j * m.ky2d * psi_hat,
            1j * m.kx2d * psi_hat,
        ), axis=0)
        v_hat = cp.stack((
            -1j * m.ky2d * llv_hat,
            1j * m.kx2d * llv_hat,
        ), axis=0)
        strain_hat = cp.stack((
            m.kx2d * m.ky2d * psi_hat,
            0.5 * (m.ky2d**2 - m.kx2d**2) * psi_hat,
        ), axis=0)
        omega_hat = 0.5 * m.kk**2 * psi_hat
        grad_v_hat = cp.stack((
            1j * m.kx2d * v_hat[0],
            1j * m.ky2d * v_hat[0],
            1j * m.kx2d * v_hat[1],
            1j * m.ky2d * v_hat[1],
        ), axis=0)

        velocity_pad = self._padded_real(v_hat)
        strain_pad = self._padded_real(strain_hat)
        grad_v_pad = self._padded_real(grad_v_hat)
        vpx, vpy = velocity_pad[0], velocity_pad[1]
        speed2_pad = vpx * vpx + vpy * vpy
        denominator = cp.mean(speed2_pad)
        if float(denominator.get()) <= 0.0:
            raise ValueError("Runtime CLE transfer diagnostic has zero velocity norm.")
        total_strain = cp.hypot(strain_pad[0], strain_pad[1])
        total_vsv = (
            strain_pad[0] * (vpx * vpx - vpy * vpy)
            + 2.0 * strain_pad[1] * vpx * vpy
        )
        p_plus = 0.5 * cp.mean(
            total_strain * speed2_pad - total_vsv
        ) / denominator
        p_minus = 0.5 * cp.mean(
            total_strain * speed2_pad + total_vsv
        ) / denominator

        spectral_l2 = cp.sum(
            cp.abs(v_hat[0])**2 + cp.abs(v_hat[1])**2,
            dtype=cp.float64,
        )
        p_strain_matrix = cp.empty(
            (n_shells, n_shells), dtype=cp.float64
        )
        p_velocity_matrix = cp.empty_like(p_strain_matrix)

        for source_index in range(n_shells):
            source_mask = shell_id == source_index
            u_hat_m = cp.where(source_mask[None, :, :], u_hat, 0.0)
            strain_hat_m = cp.where(
                source_mask[None, :, :], strain_hat, 0.0
            )
            omega_hat_m = cp.where(source_mask, omega_hat, 0.0)

            u_m_pad = self._padded_real(u_hat_m)
            strain_m_pad = self._padded_real(strain_hat_m)
            omega_m_pad = self._padded_real(omega_hat_m)
            sm11, sm12 = strain_m_pad[0], strain_m_pad[1]

            strain_x = sm11 * vpx + sm12 * vpy
            strain_y = sm12 * vpx - sm11 * vpy
            rotation_x = omega_m_pad * vpy
            rotation_y = -omega_m_pad * vpx
            advection_x = (
                u_m_pad[0] * grad_v_pad[0]
                + u_m_pad[1] * grad_v_pad[1]
            )
            advection_y = (
                u_m_pad[0] * grad_v_pad[2]
                + u_m_pad[1] * grad_v_pad[3]
            )
            full_x = strain_x + rotation_x + advection_x
            full_y = strain_y + rotation_y + advection_y

            strain_action_hat = self._unpad_fft(
                cp.stack((strain_x, strain_y), axis=0)
            )
            if m.alpha == 2 and not m.gamma:
                full_action_hat = self._unpad_fft(
                    cp.stack((full_x, full_y), axis=0)
                )
            else:
                # For generalized inversion the velocity tendency follows
                # the active-scalar tangent equation; the NSE strain identity is
                # retained above as a separate geometric diagnostic.
                psi_source = cp.where(source_mask, psi_hat, 0.0)
                q_source = cp.where(source_mask, m.q_hat, 0.0)
                jac_source = (
                    m._compute_jacobian(psi_source, llv_q_hat)
                    + m._compute_jacobian(llv_hat, q_source)
                )
                action_psi = m.inversion * jac_source
                full_action_hat = cp.stack((
                    -1j * m.ky2d * action_psi,
                    1j * m.kx2d * action_psi,
                ), axis=0)
            strain_modal = -cp.real(
                cp.conj(v_hat[0]) * strain_action_hat[0]
                + cp.conj(v_hat[1]) * strain_action_hat[1]
            )
            velocity_modal = -cp.real(
                cp.conj(v_hat[0]) * full_action_hat[0]
                + cp.conj(v_hat[1]) * full_action_hat[1]
            )
            p_strain_matrix[source_index] = self._transfer_shell_sum(
                strain_modal, shell
            ) / spectral_l2
            p_velocity_matrix[source_index] = self._transfer_shell_sum(
                velocity_modal, shell
            ) / spectral_l2

        result = {
            "P_strain_source_receiver": cp.asnumpy(p_strain_matrix),
            "P_velocity_source_receiver": cp.asnumpy(p_velocity_matrix),
            "P_plus": float(p_plus.item()),
            "P_minus": float(p_minus.item()),
        }
        return result

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
        self.l2norm_var = self.dds.variables['l2norm']
        self.evk_var = self.dds.variables['evk']
        self.evk_var.description = 'generalized error energy -0.5*Re[conj(epsi)*eq], velocity-L2 normalized'
        self.zvk_var = self.dds.variables['zvk']
        self.pvk_var = self._bind_diag_var(
            'pvk', 'f8', ('time', 'k'),
            'velocity-L2-normalized LLV PV-gradient palinstrophy spectrum; '
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
            'tpadvk', 'f8', ('time', 'k'),
            'signed advective tendency of velocity-L2-normalized LLV '
            'PV-gradient palinstrophy, '
            '-k_mode^2*Re[conj(eq)*J(psi_ref,eq)], weighted before the '
            'complete-radial-shell sum')
        self.tpprodk_var = self._bind_diag_var(
            'tpprodk', 'f8', ('time', 'k'),
            'signed production tendency of velocity-L2-normalized LLV '
            'PV-gradient palinstrophy, '
            '-k_mode^2*Re[conj(eq)*J(epsi,q_ref)], weighted before the '
            'complete-radial-shell sum')
        self.tpvisck_var = self._bind_diag_var(
            'tpvisck', 'f8', ('time', 'k'),
            'signed hyperviscous tendency of velocity-L2-normalized LLV '
            'PV-gradient palinstrophy, '
            'k_mode^2*Re[conj(eq)*(hylap*eq)], weighted before the '
            'complete-radial-shell sum')
        self.tprodk_var = self._bind_diag_var(
            'tprodk', 'f8', ('time', 'k'),
            'combined generalized-energy production spectrum teadvk + teprodk '
            'on complete radial shells k < N/2')
        self.tdissk_var = self._bind_diag_var(
            'tdissk', 'f8', ('time', 'k'),
            'combined energy dissipation tendency tefrick + tevisck '
            'on complete radial shells k < N/2')
        for name, description in (
                ('kvk', 'kinetic error energy: 0.5*k^2*|epsi|^2'),
                ('tkadvk', 'kinetic error advection tendency'),
                ('tkprodk', 'kinetic error production from J(epsi,q_ref)'),
                ('tkfrick', 'kinetic error friction tendency'),
                ('tkvisck', 'kinetic error viscous tendency'),
                ('tkproductionk', 'combined kinetic production: tkadvk + tkprodk'),
                ('tkdissk', 'combined kinetic dissipation tendency: tkfrick + tkvisck')):
            setattr(self, name + '_var', self._bind_diag_var(
                name, 'f8', ('time', 'k'),
                'velocity-L2-normalized ' + description + '; complete radial shells'))
        self.prodi_var = self._bind_diag_var(
            'prod_i', 'f8', ('time',),
            'instantaneous full-FFT-square kinetic production, velocity-L2 normalized')
        self.dissi_var = self._bind_diag_var(
            'diss_i', 'f8', ('time',),
            'instantaneous full-FFT-square positive kinetic dissipation')
        self.lbudgeti_var = self._bind_diag_var(
            'lam_budget_i', 'f8', ('time',),
            'instantaneous prod_i - diss_i, comparable to local lambda')
        self.lbudget_residi_var = self._bind_diag_var(
            'lam_budget_resid_i', 'f8', ('time',),
            'finite-interval lam_i minus endpoint lam_budget_i; not an exact closure residual')
        self.prod_var = self._bind_diag_var(
            'prod', 'f8', ('time',), 'running mean of prod_i')
        self.diss_var = self._bind_diag_var(
            'diss', 'f8', ('time',), 'running mean of diss_i')
        self.lbudget_var = self._bind_diag_var(
            'lam_budget', 'f8', ('time',), 'running mean of lam_budget_i')
        self.lbudget_resid_var = self._bind_diag_var(
            'lam_budget_resid', 'f8', ('time',), 'running lam - lam_budget')
        ds = self.dds
        shell = self._transfer_shell_contract()
        for prefix in ("source", "receiver"):
            dimension = f"{prefix}_shell"
            if dimension not in ds.dimensions:
                ds.createDimension(dimension, shell["n_shells"])
                ds.createVariable(f"{prefix}_shell_index", "i4", (dimension,))[:] = shell["index"]
                ds.createVariable(f"{prefix}_shell_k", "f8", (dimension,),
                                  fill_value=np.nan)[:] = shell["k"]
                ds.createVariable(f"{prefix}_shell_is_outer_residual", "i1",
                                  (dimension,))[:] = shell["is_outer_residual"]
        if "shell_edge" not in ds.dimensions:
            ds.createDimension("shell_edge", shell["n_shells"] + 1)
            ds.createVariable("shell_edge", "f8", ("shell_edge",))[:] = shell["edge"]
        for name in TRANSFER_NAMES:
            dims = (("time",) if name in ("P_plus", "P_minus") else
                    ("time", "source_shell", "receiver_shell"))
            for suffix in ("_i", ""):
                key = name + suffix
                description = (
                    "Instantaneous " + name if suffix else
                    "Cumulative mean of " + name + "_i over the same samples as lam")
                setattr(self, key + "_var", self._bind_diag_var(key, "f8", dims, description))

    def _stamp_output_metadata(self, ds):
        """Add model and spectral conventions to CLE diagnostic/snapshot files."""
        m = self.m_ref
        m._write_model_metadata(ds)
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
        ds.precision = m.precision
        ds.rescale_norm = "velocity_l2"
        ds.error_normalization = "velocity_l2"
        ds.budget_revision = BUDGET_REVISION
        ds.velocity_l2_target = 1.0
        ds.perturbation_kinetic_energy_target = 0.5
        if 'perturbation_energy_target' in ds.ncattrs():
            ds.delncattr('perturbation_energy_target')
        ds.normalization = (
            'llv = delta_psi/||delta||_2, with ||delta||_2^2 = '
            '(Nx*Ny)^-2 sum_modes k^2*|delta_psi_hat|^2 = <|v|^2> = 1; '
            'the normalized kinetic error energy is 1/2; evk is '
            'generalized energy -<epsi*eq>/2 and kvk is kinetic energy; spectral arrays are '
            '(Nx*Ny)^-2 sums of their stated modal densities'
        )
        ds.shell_convention = (
            'rounded radial grid-index shells; k = shell_index*(2*pi/Lx); '
            'save shell_index < Nx/2; every k-dependent density weight uses '
            'the exact modal wavenumber before shell aggregation'
        )

    def _validate_output_metadata(self, ds):
        """Never relabel historical budgets while opening an append file."""
        self.m_ref._validate_model_metadata(ds)
        if (getattr(ds, 'error_normalization', '') != 'velocity_l2'
                or getattr(ds, 'budget_revision', '') != BUDGET_REVISION):
            raise ValueError('CLE output has a different or unspecified error-budget convention; '
                             'use a new output directory.')

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
            try:
                self._validate_output_metadata(self.dds)
            except ValueError:
                self.dds.close()
                self.dds = None
                raise
            self._adopt_diag_epsilon(self.dds, nc_filename)
            self._bind_diag_vars()
        else:
            self.dds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            self.dds.createDimension('time', None)
            self.dds.createDimension('k', len(self.m_ref.kk_iso))
            kk = self.dds.createVariable('k', 'f8', ('k',))
            kk[:] = self.m_ref.kk_iso.get()
            self.d_times = self.dds.createVariable('time', 'f8', ('time',))
            self.lami_var = self.dds.createVariable('lam_i', 'f8', ('time',))
            self.lam_var = self.dds.createVariable('lam', 'f8', ('time',))
            self.l2norm_var = self.dds.createVariable('l2norm', 'f8', ('time',))
            self.evk_var = self.dds.createVariable('evk', 'f8', ('time', 'k'))
            self.zvk_var = self.dds.createVariable('zvk', 'f8', ('time', 'k'))
            self.teadvk_var = self.dds.createVariable('teadvk', 'f8', ('time', 'k'))
            self.teprodk_var = self.dds.createVariable('teprodk', 'f8', ('time', 'k'))
            self.tefrick_var = self.dds.createVariable('tefrick', 'f8', ('time', 'k'))
            self.tevisck_var = self.dds.createVariable('tevisck', 'f8', ('time', 'k'))
            self.tzadvk_var = self.dds.createVariable('tzadvk', 'f8', ('time', 'k'))
            self.tzprodk_var = self.dds.createVariable('tzprodk', 'f8', ('time', 'k'))
            self.tzfrick_var = self.dds.createVariable('tzfrick', 'f8', ('time', 'k'))
            self.tzvisck_var = self.dds.createVariable('tzvisck', 'f8', ('time', 'k'))
            self._bind_diag_vars()
            self.dds.description = "QG CLE/conditional-LLV diagnostics (master-slave, rescaled)"
            self.dds.precision = self.m_ref.precision
            self.dds.Nobs = self.Nobs
            self.dds.dTobs = self.dTobs
            self.dds.dT_cle = self.dT
            self.dds.epsilon = self.epsilon
            self.dds.rescale_norm = "velocity_l2"
            self.dds.seed = self.seed
            self.dds.dt = self.dt
            self.dds.file_index = nf
        # Append files have already passed convention validation.
        self._stamp_output_metadata(self.dds)

    def _init_lam_sums(self, n_diag_done, nsave, prefix='cle_d'):
        self._lam_sum = 0.0
        self._lam_n = 0
        self._prod_sum = 0.0
        self._diss_sum = 0.0
        self._lbudget_sum = 0.0
        n_shells = self._transfer_shell_contract()["n_shells"]
        self._transfer_sums = {
            name: (0.0 if name in ("P_plus", "P_minus") else
                   np.zeros((n_shells, n_shells), dtype=np.float64))
            for name in TRANSFER_NAMES
        }

        remaining = int(n_diag_done)
        nf = 0
        while remaining > 0:
            nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
            if not os.path.exists(nc_filename):
                print(f"[create_diag_nc] missing previous diagnostic file {nc_filename}; "
                      "running lam averages restart from available records")
                break
            with nc.Dataset(nc_filename, 'r', format='NETCDF4') as ds:
                self._validate_output_metadata(ds)
                if hasattr(ds, 'epsilon'):
                    self._adopt_diag_epsilon(ds, nc_filename)
                n_take = min(remaining, nsave, ds.variables['lam_i'].shape[0])
                if n_take <= 0:
                    break
                prev = np.asarray(ds.variables['lam_i'][:n_take], dtype=np.float64)
                previous_transfer = {
                    name: np.asarray(np.ma.filled(ds[name][n_take - 1], np.nan), dtype=np.float64)
                    for name in TRANSFER_NAMES
                }
                if 'prod_i' in ds.variables and 'diss_i' in ds.variables:
                    prev_prod = np.asarray(ds.variables['prod_i'][:n_take], dtype=np.float64)
                    prev_diss = np.asarray(ds.variables['diss_i'][:n_take], dtype=np.float64)
                elif 'tprodk' in ds.variables and 'tdissk' in ds.variables:
                    prev_prod = np.nansum(ds.variables['tprodk'][:n_take], axis=1)
                    prev_diss = -np.nansum(ds.variables['tdissk'][:n_take], axis=1)
                else:
                    prev_prod = np.full(n_take, np.nan)
                    prev_diss = np.full(n_take, np.nan)
            self._lam_sum += float(np.nansum(prev))
            self._lam_n += int(np.isfinite(prev).sum())
            self._prod_sum += float(np.nansum(prev_prod))
            self._diss_sum += float(np.nansum(prev_diss))
            self._lbudget_sum += float(np.nansum(prev_prod - prev_diss))
            # Restore only the last cumulative matrices, avoiding a read of
            # every historical instantaneous source-receiver matrix.
            self._transfer_sums = {
                name: value * self._lam_n for name, value in previous_transfer.items()
            }
            remaining -= n_take
            nf += 1

    def save_diag(self, it, t_abs, lam_i, l2norm, budget, transfer):
        (evk, zvk, pvk, teadvk, teprodk, tefrick, tevisck,
         tzadvk, tzprodk, tzfrick, tzvisck,
         tpadvk, tpprodk, tpvisck, prod_i, diss_i,
         kvk, tkadvk, tkprodk, tkfrick, tkvisck) = budget
        tprodk = teadvk + teprodk
        tdissk = tefrick + tevisck
        lam_budget_i = prod_i - diss_i
        self.d_times[it] = t_abs
        self.lami_var[it] = lam_i
        self._lam_sum += lam_i
        self._lam_n += 1
        running_lam = self._lam_sum / self._lam_n
        self.lam_var[it] = running_lam
        self.l2norm_var[it] = l2norm
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
        for name, values in (('kvk', kvk), ('tkadvk', tkadvk),
                             ('tkprodk', tkprodk), ('tkfrick', tkfrick),
                             ('tkvisck', tkvisck),
                             ('tkproductionk', tkadvk + tkprodk),
                             ('tkdissk', tkfrick + tkvisck)):
            getattr(self, name + '_var')[it, :] = values
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
        for name in TRANSFER_NAMES:
            value = transfer[name]
            getattr(self, name + "_i_var")[it] = value
            self._transfer_sums[name] += value
            getattr(self, name + "_var")[it] = self._transfer_sums[name] / self._lam_n
        self.dds.sync()

    def _bind_snapshot_var(self, name, description):
        if name in self.ds.variables:
            var = self.ds.variables[name]
        else:
            var = self.ds.createVariable(name, 'f8', ('time', 'y', 'x'), zlib=False)
        var.description = description
        return var

    def create_nc(self, nf, prefix='cle_o'):
        """Snapshot file with master q, raw perturbations, and normalized LLV."""
        nc_filename = os.path.join(self.savedir, "%s_%04d.nc" % (prefix, nf))
        if os.path.exists(nc_filename) and not self.is_not_rst:
            self.ds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            try:
                self._validate_output_metadata(self.ds)
            except ValueError:
                self.ds.close()
                raise
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
            self.times = self.ds.createVariable('time', 'f8', ('time',))
            xs = self.ds.createVariable('x', 'f8', ('x',))
            ys = self.ds.createVariable('y', 'f8', ('y',))
            xs[:] = self.m_ref.x.get()
            ys[:] = self.m_ref.y.get()
            self.q_var = self.ds.createVariable('q', 'f8', ('time', 'y', 'x'), zlib=False)
            self.llv_var = self.ds.createVariable('llv', 'f8', ('time', 'y', 'x'), zlib=False)
            self.delta_psi_var = self._bind_snapshot_var('delta_psi', 'raw delta_psi')
            self.delta_q_var = self._bind_snapshot_var('delta_q', 'raw delta_q')
            self.ds.description = "QG CLE run snapshots"
            self.ds.Nobs = self.Nobs
            self.ds.dT_cle = self.dT
            self.ds.epsilon = self.epsilon
            self.ds.rescale_norm = "velocity_l2"
            self.ds.llv_description = "delta_psi normalized by the velocity L2 norm"
        self._stamp_output_metadata(self.ds)

    def save_var(self, it):
        self.times[it] = self.m_ref.t
        self.q_var[it, :, :] = ifft2(self.m_ref.q_hat).real.get()
        dq = self._delta_q()
        dpsi = self._delta_psi(dq)
        l2norm = self._l2norm(dpsi)
        dpsi_r = ifft2(dpsi).real
        dq_r = ifft2(dq).real
        self.delta_psi_var[it, :, :] = dpsi_r.get()
        self.delta_q_var[it, :, :] = dq_r.get()
        self.llv_var[it, :, :] = (dpsi_r / max(l2norm, 1e-30)).get()
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
        ds.cle_rescale_norm = "velocity_l2"
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
        if tsave % self.intvl != 0:
            raise ValueError("tsave must be a multiple of the rescaling interval "
                             f"({self.intvl} steps) so snapshots align with CLE measurements.")

        total_steps = int(round(tmax / self.dt))
        nrst = nsave
        ndiag = max(1, int(round((tsave * nsave) / self.intvl)))
        print(f"Starting CLE run. Nobs={self.Nobs}, dTobs={self.dTobs}, "
              f"dT_cle={self.dT}, epsilon={self.epsilon:.6e}, "
              f"rescale_norm=velocity_l2, tmax={tmax}")

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
        self._l2norm_prev = self._l2norm(dpsi0)

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
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}      "
                      f"step {n:7d}      t={self.m_ref.t:9.6f}s      E_ref={E_r:.4e}      "
                      f"lam={lam_e:.4e}")

            # measure and rescale at the end of each dT_cle interval
            if n > n_start and n % self.intvl == 0:
                dq = self._delta_q()
                dpsi = self._delta_psi(dq)
                l2norm = self._l2norm(dpsi)
                lam_i = np.log(l2norm / max(self._l2norm_prev, 1e-300)) / self.dT
                budget = self._err_budget(dpsi, l2norm)
                transfer = self._compute_transfer_snapshot(dpsi, l2norm)
                if indiag == ndiag:
                    itdiag = 0
                    if nfdiag > nfdiag0:
                        self.close_diag()
                    self.create_diag_nc(nfdiag)
                    indiag = 0
                    nfdiag += 1
                self.save_diag(itdiag, float(self.m_ref.t), lam_i, l2norm, budget, transfer)
                itdiag += 1
                indiag += 1
                fac = self.epsilon / max(l2norm, 1e-300)
                self._rescale(fac)
                # Rescaling is uniform in the velocity-L2 norm.
                self._l2norm_prev = l2norm * fac

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

            if self._reference_forcing:
                ref_rk4 = scheme == 'rk4' or (self.m_ref.is_not_rst and n < 2)
                cle_rk4 = scheme == 'rk4' or (self.m_cle.is_not_rst and n < 2)
                if ref_rk4 != cle_rk4:
                    raise ValueError('Reference-normalized forcing requires matching RK4/AB3 stages.')
                forcing_stages = []
                self.m_ref._step_forward(forcing_out=forcing_stages)
                self.m_cle._step_forward(forcing_stages=forcing_stages)
            else:
                self.m_ref._step_forward()
                self.m_cle._step_forward()
            self.m_ref.t = self.m_ref.trst + (n + 1) * self.dt
            self.m_cle.t = self.m_cle.trst + (n + 1) * self.dt

        self.close_nc()
        self.close_rst()
        self.close_diag()
        print('Done.')
