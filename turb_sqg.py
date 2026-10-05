"""Ocean surface quasi-geostrophic DNS, following turb2d.py's model layout.

The surface is z=0, the ocean is z<=0, f0>0 and N0 is constant. Interior
PV anomaly is zero. Physical buoyancy b [m s^-2] obeys

    b_t + J(psi,b) = F_b + hyvisc*laplacian(b) - friction*P_L(b),
    psi_hat = b_hat/(N0*k),  u=-psi_y,  v=psi_x.

Constructor arguments, initial fields, time, coordinates and public diagnostic
outputs use SI units. Spectral states b_hat/p_hat, derivative operators and
RHS histories are nondimensional, with L_ref, U_ref, B_ref=N0*U_ref and
T_ref=L_ref/U_ref. Diagnostic methods accept these internal spectral arrays.
E denotes column-integrated QG energy per horizontal area and reference
density [m^3 s^-2]; B is half the surface buoyancy variance [m^2 s^-4].

For comparison with turb2d.py's alpha=1 convention, q=-b/B_ref, not b/B_ref:
psi_hat=-q_hat/k, while relative vorticity is k*q_hat. Use
get_comparison_state() for an explicit nondimensional export. The physical
buoyancy interface and its horizontal-mean diagnostic normalization remain
unchanged; turb2d.py's grid-sum normalization is available in that export.

The full FFT, 3/2 padding, Jacobian, RK4/AB3 and NetCDF method organization
follow turb2d.py. CuPy is used when a CUDA device is available; NumPy is a
CPU fallback. No process-wide SciPy FFT backend is changed. netCDF4 and
matplotlib are needed only for file output and plotting, respectively.

Example (all dimensional parameters):
    model = SQGModel(256, 256, hyvisc=5., forcing='markov')
    model.set_initial_condition('gauss', b_rms=4.e-5)
    model.run(tmax=86400., savedir='sqg_run')
"""

import json
import os

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np
else:
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            cp = np
    except cp.cuda.runtime.CUDARuntimeError:
        cp = np

if cp is np:
    try:
        from scipy.fft import fft2, ifft2
        FFT_BACKEND = 'scipy.fft'
    except ImportError:
        from numpy.fft import fft2, ifft2
        FFT_BACKEND = 'numpy.fft'
else:
    from cupyx.scipy.fft import fft2, ifft2
    FFT_BACKEND = 'cupyx.scipy.fft'


def _host(value):
    return cp.asnumpy(value) if cp is not np else np.asarray(value)


def _netcdf():
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ImportError('SQG NetCDF output requires netCDF4>=1.7.') from exc
    return nc


class SQGModel:
    """Constant-stratification SQG with ordinary horizontal diffusion only."""

    def __init__(self, Nx, Ny, Lx=400.e3, Ly=400.e3, dt=60.,
                 f0=1.e-4, N0=4.e-3,
                 friction=0., k_friction=None, hyvisc=1.,
                 forcing=None, fscale=None, fwidth=None, finput=None,
                 famp=1.e-8, t_r=3600., forcing_norm='amplitude',
                 precision='single', L_ref=None, U_ref=0.1,
                 z_levels=(0., -100., -500.), seed=10):
        """Initialize the grid, operators and forcing.

        Nx, Ny: even grid sizes >=8, stored in (Ny,Nx) order.
        Lx, Ly, L_ref: lengths [m]; default L_ref=Lx/(2*pi).
        dt, t_r: timestep and Markov correlation time [s], both positive.
        f0, N0: positive Coriolis and buoyancy frequencies [s^-1].
        U_ref: fixed velocity scale [m s^-1], independent of evolving RMS.
        hyvisc: buoyancy diffusivity [m^2 s^-1]; the familiar name is retained
            from turb2d.py, but this DNS implementation is Laplacian only.
        friction: nonnegative drag [s^-1] on 0<k<=k_friction [rad m^-1].
        fscale, fwidth: center and full width of the forcing annulus
            [rad m^-1]; defaults are 4 and 4 times 2*pi/Lx, respectively,
            matching turb2d.py's center 4 and half-width 2.
        famp: stationary ensemble RMS of the unnormalized Markov buoyancy
            tendency [m s^-3]. Forcing starts from zero memory.
        forcing: None or 'markov'. A zero-correlation-time white-noise limit
            needs a different timestep normalization and is not used here.
        forcing_norm: 'amplitude' preserves the prescribed Markov process.
            'energy'/'buoyancy' add a state-dependent correction to impose
            beginning-of-step E/B input finput [m^3 s^-3]/[m^2 s^-5]. This is
            an affine projection, not Smith et al.'s random-overlap division.
            Injection modes require nonzero state in the forcing band. The
            finite-step integrated input still has time-discretization error.
        finput: required for injection modes, unused/None for amplitude mode.
        z_levels: diagnostic depths [m], nonpositive. No vertical PDE grid.
        seed: independent reproducible forcing RNG seed; initialization uses
            a separate RNG, so changing the initial field does not change F.

        Public spectra are shell sums of horizontal means, including corner
        modes. Unlike turb2d.py's legacy grid sums, do not divide by Nx*Ny
        again. Call set_initial_condition() before a nontrivial decay run.
        """
        for name, n in (('Nx', Nx), ('Ny', Ny)):
            if isinstance(n, bool) or int(n) != n or n < 8 or n % 2:
                raise ValueError(f'{name} must be an even integer >=8.')
        for name, val in (('Lx', Lx), ('Ly', Ly), ('dt', dt), ('f0', f0),
                          ('N0', N0), ('U_ref', U_ref), ('t_r', t_r)):
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name, val in (('hyvisc', hyvisc), ('friction', friction), ('famp', famp)):
            if not np.isfinite(val) or val < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if precision not in ('single', 'double'):
            raise ValueError("precision must be 'single' or 'double'.")
        if forcing not in (None, 'markov'):
            raise ValueError("forcing must be None or 'markov'.")
        if forcing_norm not in ('amplitude', 'energy', 'buoyancy'):
            raise ValueError('Unknown forcing_norm.')
        if forcing_norm == 'amplitude' and finput is not None:
            raise ValueError('Choose energy/buoyancy forcing_norm to specify finput.')
        if forcing_norm != 'amplitude':
            if forcing != 'markov' or finput is None or not np.isfinite(finput) or finput < 0:
                raise ValueError('Injection modes require markov forcing and finput>=0.')
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.Lx, self.Ly, self.dt = float(Lx), float(Ly), float(dt)
        self.f0, self.N0 = float(f0), float(N0)
        self.L_ref = self.Lx/(2*np.pi) if L_ref is None else float(L_ref)
        if not np.isfinite(self.L_ref) or self.L_ref <= 0:
            raise ValueError('L_ref must be finite and positive.')
        self.U_ref = float(U_ref)
        self.B_ref = self.N0*self.U_ref
        self.T_ref = self.L_ref/self.U_ref
        self.Psi_ref = self.U_ref*self.L_ref
        self.H_ref = self.f0*self.L_ref/self.N0
        self.W_ref = self.U_ref**2/(self.N0*self.L_ref)
        self.E_ref = self.U_ref**2*self.H_ref
        self.dt_nd = self.dt/self.T_ref
        self.friction, self.hyvisc = float(friction), float(hyvisc)
        k_box = 2*np.pi/self.Lx
        self.k_friction = 2*k_box if k_friction is None else float(k_friction)
        self.fscale = 4*k_box if fscale is None else float(fscale)
        self.fwidth = 4*k_box if fwidth is None else float(fwidth)
        if not np.isfinite(self.k_friction) or self.k_friction < 0:
            raise ValueError('k_friction must be finite and nonnegative.')
        if not np.isfinite(self.fscale) or self.fscale <= 0:
            raise ValueError('fscale must be finite and positive.')
        if not np.isfinite(self.fwidth) or self.fwidth <= 0:
            raise ValueError('fwidth must be finite and positive.')
        self.forcing, self.forcing_norm = forcing, forcing_norm
        self.finput = None if finput is None else float(finput)
        self.famp, self.t_r = float(famp), float(t_r)
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError('seed must be a nonnegative integer.')
        self.seed = int(seed)
        # Host RNG state has a portable JSON representation for exact restart.
        # Only independent forced modes are generated/transferred, not a grid.
        self._rng = np.random.default_rng(self.seed)
        self.precision = precision
        self.rdtype = cp.float32 if precision == 'single' else cp.float64
        self.cdtype = cp.complex64 if precision == 'single' else cp.complex128
        self.backend = 'numpy' if cp is np else 'cupy'
        self.z_levels = self._check_depths(z_levels)
        self._init_grid()
        self._prebuild_operator()
        self.b_hat = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self.p_hat = cp.zeros_like(self.b_hat)
        self.rv_hat = cp.zeros_like(self.b_hat)
        self.k1_p = cp.zeros_like(self.b_hat)
        self.k1_pp = cp.zeros_like(self.b_hat)
        self._init_markovforce()
        self.ts_scheme = 'ab3'
        self.t, self.n_steps, self.history_count = 0., 0, 0
        self.is_not_rst = True

### private term
    @staticmethod
    def _check_depths(z):
        depths = np.atleast_1d(np.asarray(z, dtype=float))
        if depths.ndim != 1 or depths.size == 0 or not np.all(np.isfinite(depths)) or np.any(depths > 0):
            raise ValueError('Depths must be a nonempty vector of finite z<=0 [m].')
        return depths

    def _init_grid(self):
        """Physical coordinates and nondimensional full-FFT wavenumbers."""
        self.x = cp.arange(self.Nx, dtype=self.rdtype)*(self.Lx/self.Nx)
        self.y = cp.arange(self.Ny, dtype=self.rdtype)*(self.Ly/self.Ny)
        self.x2d, self.y2d = cp.meshgrid(self.x, self.y)
        nx = cp.asarray(np.fft.fftfreq(self.Nx)*self.Nx, dtype=self.rdtype)
        ny = cp.asarray(np.fft.fftfreq(self.Ny)*self.Ny, dtype=self.rdtype)
        self.nx2d, self.ny2d = cp.meshgrid(nx, ny)
        kx = nx*(2*np.pi*self.L_ref/self.Lx)
        ky = ny*(2*np.pi*self.L_ref/self.Ly)
        self.kx2d, self.ky2d = cp.meshgrid(kx, ky)
        self.kk = cp.sqrt(self.kx2d**2+self.ky2d**2)
        # Define physical bands independently of the arbitrary reference
        # scale. Dividing kk by L_ref can move a band-edge mode across its
        # threshold through roundoff, changing both forcing support and RNG
        # consumption when only the nondimensionalization is changed.
        kx_dim, ky_dim = cp.meshgrid(nx*(2*np.pi/self.Lx), ny*(2*np.pi/self.Ly))
        self.kk_dim = cp.sqrt(kx_dim**2+ky_dim**2)
        self.k_spacing = min(2*np.pi/self.Lx, 2*np.pi/self.Ly)
        self.kk_idx = cp.rint(self.kk_dim/self.k_spacing).astype(cp.int32)
        self.n_shells = int(_host(self.kk_idx.max()))+1
        self.kk_iso = np.arange(self.n_shells)*self.k_spacing
        self.spectral_mask = cp.ones((self.Ny, self.Nx), dtype=bool)
        self.spectral_mask[:, self.Nx//2] = False
        self.spectral_mask[self.Ny//2, :] = False
        self.spectral_mask[0, 0] = False

    def _init_friction(self):
        self.friction_mask = ((self.kk_dim <= self.k_friction)
                              & self.spectral_mask).astype(self.rdtype)

    def _prebuild_operator(self):
        self.inversion = cp.zeros_like(self.kk)
        nonzero = self.kk > 0
        self.inversion[nonzero] = 1/self.kk[nonzero]
        self.lap = -self.kk**2
        self.hylap = (self.hyvisc*self.T_ref/self.L_ref**2)*self.lap
        self._init_friction()
        self.linear_damping = self.hylap-self.friction*self.T_ref*self.friction_mask
        self.Nxpad, self.Nypad = 3*self.Nx//2, 3*self.Ny//2
        self.pad_buffer = cp.zeros((self.Nypad, self.Nxpad), dtype=self.cdtype)
        self.parseval_fac = (self.Nxpad*self.Nypad)/(self.Nx*self.Ny)

    def _padding(self, ft):
        """3/2 padding in FFT order, with the even-grid Nyquist axes omitted."""
        self.pad_buffer.fill(0j)
        hx, hy = self.Nx//2, self.Ny//2
        sx, sy = slice(hx+1, self.Nx), slice(hy+1, self.Ny)
        px, py = slice(self.Nxpad-hx+1, self.Nxpad), slice(self.Nypad-hy+1, self.Nypad)
        self.pad_buffer[:hy, :hx] = ft[:hy, :hx]*self.parseval_fac
        self.pad_buffer[:hy, px] = ft[:hy, sx]*self.parseval_fac
        self.pad_buffer[py, :hx] = ft[sy, :hx]*self.parseval_fac
        self.pad_buffer[py, px] = ft[sy, sx]*self.parseval_fac
        return self.pad_buffer

    def _unpadding(self, ft_pad):
        hx, hy = self.Nx//2, self.Ny//2
        sx, sy = slice(hx+1, self.Nx), slice(hy+1, self.Ny)
        px, py = slice(self.Nxpad-hx+1, self.Nxpad), slice(self.Nypad-hy+1, self.Nypad)
        ft = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        ft[:hy, :hx] = ft_pad[:hy, :hx]/self.parseval_fac
        ft[:hy, sx] = ft_pad[:hy, px]/self.parseval_fac
        ft[sy, :hx] = ft_pad[py, :hx]/self.parseval_fac
        ft[sy, sx] = ft_pad[py, px]/self.parseval_fac
        return self._enforce_spectral_constraints(ft)

    def _enforce_spectral_constraints(self, ft):
        """Remove mean and Nyquist axes; inputs must already be Hermitian."""
        ft[~self.spectral_mask] = 0.
        return ft

    def _compute_jacobian(self, p_hat, b_hat):
        """J(psi,b)=psi_x*b_y-psi_y*b_x, using turb2d.py's 3/2 rule."""
        dxb = ifft2(self._padding(1j*self.kx2d*b_hat)).real
        dyb = ifft2(self._padding(1j*self.ky2d*b_hat)).real
        dxp = ifft2(self._padding(1j*self.kx2d*p_hat)).real
        dyp = ifft2(self._padding(1j*self.ky2d*p_hat)).real
        return self._unpadding(fft2(dxp*dyb-dyp*dxb))

    def _get_rhs(self, b_hat):
        p_hat = self.inversion*b_hat
        rhs = self.linear_damping*b_hat-self._compute_jacobian(p_hat, b_hat)
        rhs += self.force_b
        return self._enforce_spectral_constraints(rhs.astype(self.cdtype))

    def _rk4(self, b_hat):
        dt = self.dt_nd
        k1 = self._get_rhs(b_hat)
        k2 = self._get_rhs(b_hat+0.5*dt*k1)
        k3 = self._get_rhs(b_hat+0.5*dt*k2)
        k4 = self._get_rhs(b_hat+dt*k3)
        return k1, k2, k3, k4

    def _step_forward(self):
        """One full step with a fixed Markov sample throughout the interval.

        AB3 histories contain only advection/damping. Adding dt*F separately
        gives the same piecewise-constant forcing integral as RK4. Stochastic
        forcing does not inherit the deterministic RK4/AB3 convergence order.
        """
        if self.ts_scheme not in ('rk4', 'ab3'):
            raise ValueError("Time scheme must be 'rk4' or 'ab3'.")
        if self.forcing == 'markov':
            self._update_markovforce()
        if self.ts_scheme == 'rk4' or self.history_count < 2:
            k1, k2, k3, k4 = self._rk4(self.b_hat)
            b_new = self.b_hat+self.dt_nd/6*(k1+2*k2+2*k3+k4)
        else:
            k1 = self._get_rhs(self.b_hat)
            b_new = (self.b_hat+self.dt_nd/12*(23*(k1-self.force_b)
                     -16*self.k1_p+5*self.k1_pp)+self.dt_nd*self.force_b)
        self.k1_pp = self.k1_p.copy()
        self.k1_p = (k1-self.force_b).astype(self.cdtype)
        self.history_count = min(self.history_count+1, 2)
        self.b_hat = self._enforce_spectral_constraints(b_new.astype(self.cdtype))
        self._update_state()
        self.n_steps += 1
        self.t = self.n_steps*self.dt

    def _update_state(self):
        self.p_hat = (self.inversion*self.b_hat).astype(self.cdtype)
        self.rv_hat = (self.lap*self.p_hat).astype(self.cdtype)

### initial term
    def set_initial_condition(self, scheme='gauss', eini=None, b_ini=None,
                              trst=0., b_rms=None, kmin=None, kmax=None, seed=None):
        """Set physical surface buoyancy, then diagnose the other fields.

        'gauss': a real Gaussian random field in a finite annulus (defaults
            3<=k*Lx/(2*pi)<=5), normalized to b_rms, default 0.1*B_ref.
        'ellipse': periodic smooth elliptical Gaussian, mean removed.
        'zero': zero buoyancy (valid for amplitude forcing spin-up).
        'field': b_ini [m s^-2], shape (Ny,Nx); complex inputs are dimensional
            full-FFT coefficients of a real field. Its amplitude is retained
            unless b_rms or eini is supplied. A mean anomaly is removed.
        'rst': b_ini is a restart filename; load_rst restores full history.
        eini: optional initial column energy [m^3 s^-2], exclusive of b_rms.
        trst: physical time for a fresh field; AB3 starts again with RK4.
        """
        if scheme == 'rst':
            if not isinstance(b_ini, (str, os.PathLike)):
                raise ValueError("For scheme='rst', b_ini must be a restart filename.")
            return self.load_rst(b_ini)
        if eini is not None and b_rms is not None:
            raise ValueError('Specify either eini or b_rms.')
        for name, val in (('eini', eini), ('b_rms', b_rms)):
            if val is not None and (not np.isfinite(val) or val < 0):
                raise ValueError(f'{name} must be finite and nonnegative.')
        if not np.isfinite(trst) or trst < 0 or not np.isclose(trst/self.dt, round(trst/self.dt), rtol=0, atol=1.e-8):
            raise ValueError('trst must be a nonnegative integer multiple of dt.')
        rng = np.random.default_rng(self.seed+1 if seed is None else seed)
        if scheme == 'gauss':
            k_box = 2*np.pi/self.Lx
            kmin = 3*k_box if kmin is None else float(kmin)
            kmax = 5*k_box if kmax is None else float(kmax)
            if not np.isfinite(kmin+kmax) or not 0 <= kmin < kmax:
                raise ValueError('Initial annulus requires 0<=kmin<kmax.')
            mask = (self.kk_dim >= kmin) & (self.kk_dim <= kmax) & self.spectral_mask
            if not bool(_host(cp.any(mask))):
                raise ValueError('Initial annulus contains no retained Fourier modes.')
            noise = cp.asarray(rng.standard_normal((self.Ny, self.Nx)), dtype=self.rdtype)
            self.b_hat = (fft2(noise)*mask).astype(self.cdtype)
        elif scheme == 'ellipse':
            # Fix the physical shape to the box, independently of reference
            # scales used only to nondimensionalize the evolution.
            width_x, width_y = self.Lx/(2*np.pi), self.Ly/(2*np.pi)
            x = (self.x2d-self.Lx/2)/width_x
            y = (self.y2d-self.Ly/2)/width_y
            # Sum images, rather than truncate a nonperiodic Gaussian at edges.
            field = cp.zeros_like(self.x2d)
            for ix in range(-2, 3):
                for iy in range(-2, 3):
                    field += cp.exp(-(x+ix*2*np.pi)**2-16*(y+iy*2*np.pi)**2)
            self.b_hat = fft2(field).astype(self.cdtype)
        elif scheme == 'zero':
            self.b_hat = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        elif scheme == 'field':
            field = cp.asarray(b_ini)
            if field.shape != (self.Ny, self.Nx) or not bool(_host(cp.all(cp.isfinite(field)))):
                raise ValueError('b_ini must be a finite (Ny,Nx) field.')
            if field.dtype.kind == 'c':
                physical = ifft2(field)
                tol = 100*np.finfo(np.dtype(self.rdtype)).eps
                scale = max(float(_host(cp.max(cp.abs(physical)))), np.finfo(float).tiny)
                if float(_host(cp.max(cp.abs(physical.imag)))) > tol*scale:
                    raise ValueError('Spectral b_ini must represent a real field.')
                self.b_hat = (field/self.B_ref).astype(self.cdtype)
            else:
                self.b_hat = fft2(field.astype(self.rdtype)/self.B_ref).astype(self.cdtype)
        else:
            raise ValueError(f'Unknown initial-condition scheme: {scheme!r}.')
        self._enforce_spectral_constraints(self.b_hat)
        self._update_state()
        if scheme in ('gauss', 'ellipse') and b_rms is None and eini is None:
            b_rms = 0.1*self.B_ref
        if eini is not None:
            current, target = self.get_Etot(), float(eini)
        elif b_rms is not None:
            current, target = self.get_Btot(), 0.5*float(b_rms)**2
        else:
            current, target = 1., 1.
        if target > 0 and current <= 0:
            raise ValueError('Cannot normalize a zero field to positive amplitude.')
        self.b_hat *= 0. if target == 0 else np.sqrt(target/current)
        self._update_state()
        self.k1_p.fill(0j)
        self.k1_pp.fill(0j)
        self.force_b.fill(0j)
        self.markov_hat.fill(0j)
        self._rng = np.random.default_rng(self.seed)
        self.n_steps = int(round(trst/self.dt))
        self.t = self.n_steps*self.dt
        self.history_count = 0
        self.is_not_rst = True

### forcing term
    def _init_markovforce(self):
        self.fmask = ((self.kk_dim >= self.fscale-self.fwidth/2)
                      & (self.kk_dim <= self.fscale+self.fwidth/2)
                      & self.spectral_mask)
        n_modes = int(_host(cp.sum(self.fmask)))
        if self.forcing == 'markov' and n_modes == 0:
            raise ValueError('Forcing annulus contains no retained Fourier modes.')
        self.fR = float(np.exp(-self.dt/self.t_r))
        # Generate one complex Gaussian coefficient per conjugate pair.
        # Each pair contributes 2*E|coefficient|^2 to the full FFT Parseval
        # sum. This gives the same law as masking the FFT of real white noise
        # without allocating or transferring a full physical noise grid.
        iy, ix = np.nonzero(_host(self.fmask))
        jy, jx = (-iy) % self.Ny, (-ix) % self.Nx
        independent = iy*self.Nx+ix < jy*self.Nx+jx
        self._force_iy = cp.asarray(iy[independent])
        self._force_ix = cp.asarray(ix[independent])
        self._force_jy = cp.asarray(jy[independent])
        self._force_jx = cp.asarray(jx[independent])
        self._n_force_pairs = int(np.count_nonzero(independent))
        self.fcoef = (self.famp*self.T_ref/self.B_ref
                      *(self.Nx*self.Ny)/np.sqrt(2*max(n_modes, 1))
                      *np.sqrt(-np.expm1(-2*self.dt/self.t_r)))
        self.force_b = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self.markov_hat = cp.zeros_like(self.force_b)

    def _inner(self, a_hat, b_hat):
        """Parseval horizontal mean, reducing in double precision."""
        return float(_host(cp.sum(cp.real(cp.conj(a_hat)*b_hat), dtype=cp.float64)))/(self.Nx*self.Ny)**2

    def _update_markovforce(self):
        if self.forcing_norm != 'amplitude':
            # Gradient of the nondimensional quadratic invariant. Projection
            # uses its positive squared norm, never a random signed overlap.
            a_hat = self.fmask*(self.p_hat if self.forcing_norm == 'energy' else self.b_hat)
            denom = self._inner(a_hat, a_hat)
            if not np.isfinite(denom) or denom <= np.finfo(np.dtype(self.rdtype)).tiny:
                raise ValueError('Constant injection requires nonzero buoyancy in the forcing band; initialize a finite-band field.')
        noise = self._rng.standard_normal((2, self._n_force_pairs), dtype=np.dtype(self.rdtype))
        innovation = cp.asarray(noise[0]+1j*noise[1], dtype=self.cdtype)*self.fcoef
        self.markov_hat *= self.fR
        self.markov_hat[self._force_iy, self._force_ix] += innovation
        self.markov_hat[self._force_jy, self._force_jx] += cp.conj(innovation)
        self.force_b = self.markov_hat.copy()
        if self.forcing_norm != 'amplitude':
            rate_scale = (self.E_ref if self.forcing_norm == 'energy' else self.B_ref**2)/self.T_ref
            target = self.finput/rate_scale
            self.force_b += ((target-self._inner(a_hat, self.force_b))/denom)*a_hat
        self._enforce_spectral_constraints(self.force_b)

### diagnostic term
    def _shell_sum(self, density):
        sums = cp.bincount(self.kk_idx.ravel(), weights=density.ravel(), minlength=self.n_shells)
        return _host(sums)/(self.Nx*self.Ny)**2

    def get_Ek(self, p_hat=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        Ek = self._shell_sum(0.5*self.kk*cp.abs(p_hat)**2)
        return Ek*(self.E_ref if dimensional else 1.)

    def get_Etot(self, p_hat=None, dimensional=True):
        """Column QG energy, horizontal mean [m^3 s^-2]."""
        return float(np.sum(self.get_Ek(p_hat, dimensional)))

    def get_Bk(self, b_hat=None, dimensional=True):
        b_hat = self.b_hat if b_hat is None else b_hat
        Bk = self._shell_sum(0.5*cp.abs(b_hat)**2)
        return Bk*(self.B_ref**2 if dimensional else 1.)

    def get_Btot(self, b_hat=None, dimensional=True):
        """Half the surface buoyancy variance [m^2 s^-4]."""
        return float(np.sum(self.get_Bk(b_hat, dimensional)))

    def get_Kk(self, p_hat=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        Kk = self._shell_sum(0.5*self.kk**2*cp.abs(p_hat)**2)
        return Kk*(self.U_ref**2 if dimensional else 1.)

    def get_Ktot(self, p_hat=None, dimensional=True):
        return float(np.sum(self.get_Kk(p_hat, dimensional)))

    def get_Vrms(self, p_hat=None, dimensional=True):
        return np.sqrt(2*self.get_Ktot(p_hat, dimensional))

    def get_Brms(self, b_hat=None, dimensional=True):
        return np.sqrt(2*self.get_Btot(b_hat, dimensional))

    def get_comparison_state(self, normalization='mean'):
        """Export the instantaneous nondimensional alpha=1 comparison state.

        The generalized active tracer is q=-b/B_ref and force_q=-force_b.
        The spectral arrays are independent backend copies; modifying the
        export never changes this model. Parameters use x/L_ref, t/T_ref,
        and the same velocity/streamfunction scales as the internal solver.

        E=-<psi*q>/2, Z=<q*q>/2 and K=<|grad(psi)|^2>/2 are horizontal means
        by default. ``normalization='grid_sum'`` multiplies these invariants
        and spectra by Nx*Ny, matching turb2d.py's diagnostic convention.
        In physical units E_phys=E_ref*E_mean, B_phys=B_ref**2*Z_mean and
        K_phys=U_ref**2*K_mean. All spectra include retained corner modes;
        compare shell-by-shell only after checking both models' k arrays.

        This is a state/diagnostic export, not a restart or a conversion of
        the forcing RNG. Equal Markov amplitudes across implementations do
        not imply identical forcing samples or injection rates.
        """
        if normalization not in ('mean', 'grid_sum'):
            raise ValueError("normalization must be 'mean' or 'grid_sum'.")
        factor = self.Nx*self.Ny if normalization == 'grid_sum' else 1.
        spectra = {'Ek': factor*self.get_Ek(dimensional=False),
                   'Zk': factor*self.get_Bk(dimensional=False),
                   'Kk': factor*self.get_Kk(dimensional=False)}
        return {
            'parameters': {'Nx': self.Nx, 'Ny': self.Ny,
                           'Lx': self.Lx/self.L_ref, 'Ly': self.Ly/self.L_ref,
                           'dt': self.dt_nd, 'alpha': 1., 'beta': 0., 'gamma': 0.,
                           'precision': self.precision, 'damping_on': 'q',
                           'forcing': None,
                           'hyvisc': self.hyvisc*self.T_ref/self.L_ref**2,
                           'hyperorder': 1, 'friction': self.friction*self.T_ref,
                           'k_friction': self.k_friction*self.L_ref},
            't': self.t/self.T_ref, 'normalization': normalization,
            'q_hat': -self.b_hat, 'p_hat': self.p_hat.copy(),
            'rv_hat': self.rv_hat.copy(), 'force_q': -self.force_b,
            'k': self.kk_iso.copy()*self.L_ref,
            **spectra,
            'E': float(np.sum(spectra['Ek'])),
            'Z': float(np.sum(spectra['Zk'])),
            'K': float(np.sum(spectra['Kk'])),
        }

    def _diag_tendency(self, p_hat, b_hat, tendency, dimensional=True, nonlinear=False):
        te = self._shell_sum(cp.real(cp.conj(p_hat)*tendency))
        tb = self._shell_sum(cp.real(cp.conj(b_hat)*tendency))
        if dimensional:
            te *= self.E_ref/self.T_ref
            tb *= self.B_ref**2/self.T_ref
        if nonlinear:
            return te, tb, -np.cumsum(te), -np.cumsum(tb)
        return te, tb, np.cumsum(te[::-1])[::-1], np.cumsum(tb[::-1])[::-1]

    def get_diagNL(self, p_hat=None, b_hat=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        b_hat = self.b_hat if b_hat is None else b_hat
        tendency = -self._compute_jacobian(p_hat, b_hat)
        return self._diag_tendency(p_hat, b_hat, tendency, dimensional, nonlinear=True)

    def get_diagF(self, p_hat=None, b_hat=None, force_b=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        b_hat = self.b_hat if b_hat is None else b_hat
        force_b = self.force_b if force_b is None else force_b
        return self._diag_tendency(p_hat, b_hat, force_b, dimensional)

    def get_diagFric(self, p_hat=None, b_hat=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        b_hat = self.b_hat if b_hat is None else b_hat
        tendency = -self.friction*self.T_ref*self.friction_mask*b_hat
        return self._diag_tendency(p_hat, b_hat, tendency, dimensional)

    def get_diagVisc(self, p_hat=None, b_hat=None, dimensional=True):
        p_hat = self.p_hat if p_hat is None else p_hat
        b_hat = self.b_hat if b_hat is None else b_hat
        return self._diag_tendency(p_hat, b_hat, self.hylap*b_hat, dimensional)

    def get_w(self, z=None, dimensional=True):
        """Balanced vertical velocity [m s^-1], z<=0 supplied in metres.

        w=(D_z J_s-J_z)/N0^2 in physical variables. This diagnoses nonlinear
        balanced motion; it does not differentiate a noisy total b_t. With
        forcing/diffusion/drag it corresponds to extending the surface source
        as S(z)=D_z S_s. Other interior source choices need a correction.
        A scalar z returns (Ny,Nx); a depth vector returns (nz,Ny,Nx).
        """
        scalar = z is not None and np.ndim(z) == 0
        depths = self.z_levels if z is None else self._check_depths(z)
        w = cp.zeros((len(depths), self.Ny, self.Nx), dtype=self.rdtype)
        if np.any(depths != 0):
            js = self._compute_jacobian(self.p_hat, self.b_hat)
            for iz, depth in enumerate(depths):
                if depth == 0:
                    continue
                decay = cp.exp(self.kk*(depth/self.H_ref))
                jz = self._compute_jacobian(decay*self.p_hat, decay*self.b_hat)
                w[iz] = ifft2(decay*js-jz).real
        if dimensional:
            w *= self.W_ref
        return w[0] if scalar else w

    def get_fields(self, z=0., dimensional=True):
        """Return backend arrays b, psi, u, v, rv and w at one physical depth."""
        if np.ndim(z) != 0:
            raise ValueError('get_fields expects one depth; get_w accepts a vector.')
        depth = self._check_depths(z)[0]
        decay = cp.exp(self.kk*(depth/self.H_ref))
        p = decay*self.p_hat
        fields = {'b': ifft2(decay*self.b_hat).real,
                  'psi': ifft2(p).real,
                  'u': ifft2(-1j*self.ky2d*p).real,
                  'v': ifft2(1j*self.kx2d*p).real,
                  'rv': ifft2(self.lap*p).real,
                  'w': self.get_w(float(depth), dimensional=False)}
        scales = {'b': self.B_ref, 'psi': self.Psi_ref, 'u': self.U_ref,
                  'v': self.U_ref, 'rv': self.U_ref/self.L_ref, 'w': self.W_ref}
        return {key: (val*(scales[key] if dimensional else 1.)).astype(self.rdtype)
                for key, val in fields.items()}

    def get_cfl(self):
        """Advective CFL plus maximum explicit linear decay rate times dt."""
        u = ifft2(-1j*self.ky2d*self.p_hat).real*self.U_ref
        v = ifft2(1j*self.kx2d*self.p_hat).real*self.U_ref
        adv = self.dt*float(_host(cp.max(cp.abs(u)/(self.Lx/self.Nx)+cp.abs(v)/(self.Ly/self.Ny))))
        damp = self.dt_nd*float(_host(cp.max(-self.linear_damping*self.spectral_mask)))
        return adv, damp

    # Save and output methods
    def _config(self):
        keys = ('Nx', 'Ny', 'Lx', 'Ly', 'dt', 'f0', 'N0', 'friction',
                'k_friction', 'hyvisc', 'forcing', 'fscale', 'fwidth',
                'finput', 'famp', 't_r', 'forcing_norm', 'precision',
                'L_ref', 'U_ref', 'seed')
        config = {key: getattr(self, key) for key in keys}
        config['z_levels'] = self.z_levels.tolist()
        return config

    def _write_metadata(self, ds):
        ds.model = 'constant_N_SQG_v1'
        ds.sqg_config = json.dumps(self._config(), sort_keys=True)
        ds.ts_scheme = self.ts_scheme
        ds.backend = self.backend
        ds.fft_backend = FFT_BACKEND
        ds.description = 'Ocean SQG: dimensional output, nondimensional spectral evolution'
        for key in ('L_ref', 'U_ref', 'B_ref', 'T_ref', 'Psi_ref', 'H_ref', 'W_ref', 'E_ref'):
            ds.setncattr(key, getattr(self, key))
        ds.w_definition = 'Balanced w; assumes interior source S(z)=exp(N0*k*z/f0)*S_surface, k in rad m-1.'
        ds.forcing_time_scheme = 'One Markov sample per full step; integrated as constant over that step.'
        ds.q_definition = 'Nondimensional alpha=1 tracer q=-b/B_ref; psi_hat=-q_hat/k_nd; force_q=-force_b_nd.'
        ds.spectral_normalization = 'Horizontal means, all retained shells; no additional Nx*Ny division.'
        ds.wavenumber_convention = 'physical_grid_v1'

    def _validate_file(self, ds):
        if getattr(ds, 'model', '') != 'constant_N_SQG_v1':
            raise ValueError('Not a compatible SQG file.')
        if json.loads(ds.sqg_config) != self._config():
            raise ValueError('SQG file parameters/scales/precision do not match this model.')
        if ds.ts_scheme not in ('rk4', 'ab3'):
            raise ValueError('Invalid time scheme in SQG file.')
        convention = getattr(ds, 'wavenumber_convention', 'legacy_reference_grid')
        if convention == 'legacy_reference_grid':
            # Old files remain usable when the correction does not change
            # their active forcing/drag masks. Otherwise continuation would
            # silently switch mode support and the forcing random sequence.
            old_k = self.kk/self.L_ref
            old_force = ((old_k >= self.fscale-self.fwidth/2)
                         & (old_k <= self.fscale+self.fwidth/2)
                         & self.spectral_mask)
            old_drag = (old_k <= self.k_friction) & self.spectral_mask
            changed = ((self.forcing == 'markov'
                        and bool(_host(cp.any(old_force != self.fmask))))
                       or (self.friction > 0
                           and bool(_host(cp.any(old_drag != self.friction_mask)))))
            if changed:
                raise ValueError('Legacy SQG file uses different band-edge forcing/drag modes. '
                                 'Exact continuation requires the original solver; start a new '
                                 'run to use reference-independent physical wavenumbers.')
        elif convention != 'physical_grid_v1':
            raise ValueError('Unknown SQG physical-wavenumber convention.')

    def create_nc(self, nf, prefix='output'):
        """Create or append a dimensional diagnostic NetCDF file."""
        nc = _netcdf()
        os.makedirs(self.savedir, exist_ok=True)
        path = os.path.join(self.savedir, f'{prefix}_{nf:04d}.nc')
        exists = os.path.exists(path)
        ds = nc.Dataset(path, 'a' if exists else 'w', format='NETCDF4')
        try:
            if exists:
                self._validate_file(ds)
                if ds.ts_scheme != self.ts_scheme:
                    raise ValueError('Cannot append with a different time scheme.')
            else:
                self._write_metadata(ds)
                for name, size in (('time', None), ('x', self.Nx), ('y', self.Ny),
                                   ('z', len(self.z_levels)), ('k', self.n_shells)):
                    ds.createDimension(name, size)
                for name, data, unit in (('x', _host(self.x), 'm'), ('y', _host(self.y), 'm'),
                                         ('z', self.z_levels, 'm'), ('k', self.kk_iso, 'rad m-1')):
                    var = ds.createVariable(name, 'f8', (name,))
                    var[:] = data
                    var.units = unit
                ds.variables['z'].positive = 'up'
                ds.createVariable('time', 'f8', ('time',)).units = 's'
                dtype = 'f4' if self.precision == 'single' else 'f8'
                for name, unit in (('b', 'm s-2'), ('psi', 'm2 s-1'), ('u', 'm s-1'),
                                   ('v', 'm s-1'), ('rv', 's-1'), ('force_b', 'm s-3')):
                    ds.createVariable(name, dtype, ('time', 'y', 'x')).units = unit
                ds.variables['force_b'].description = 'Most recently applied surface buoyancy tendency.'
                ds.createVariable('w', dtype, ('time', 'z', 'y', 'x')).units = 'm s-1'
                for name, unit in (('E', 'm3 s-2'), ('B', 'm2 s-4'), ('K', 'm2 s-2')):
                    ds.createVariable(name+'tot', 'f8', ('time',)).units = unit
                    var = ds.createVariable(name+'k', 'f8', ('time', 'k'))
                    var.units = unit
                    var.description = 'Shell sum; sums to horizontal mean; not divided by shell width.'
                for process in ('nl', 'f', 'fric', 'visc'):
                    for invariant, unit in (('e', 'm3 s-3'), ('b', 'm2 s-5')):
                        for kind in ('t', 'f'):
                            name = kind+invariant+process+'k'
                            ds.createVariable(name, 'f8', ('time', 'k')).units = unit
                for name in ('cfl', 'damping_dt'):
                    ds.createVariable(name, 'f8', ('time',)).units = '1'
        except Exception:
            ds.close()
            raise
        self.ds = ds

    def save_var(self, it):
        ds = self.ds
        ds.variables['time'][it] = self.t
        fields = self.get_fields()
        for name in ('b', 'psi', 'u', 'v', 'rv'):
            ds.variables[name][it] = _host(fields[name])
        ds.variables['w'][it] = _host(self.get_w())
        ds.variables['force_b'][it] = _host(ifft2(self.force_b).real)*(self.B_ref/self.T_ref)
        for name, method in (('E', self.get_Ek), ('B', self.get_Bk), ('K', self.get_Kk)):
            spectrum = method()
            ds.variables[name+'tot'][it] = np.sum(spectrum)
            ds.variables[name+'k'][it] = spectrum
        for process, method in (('nl', self.get_diagNL), ('f', self.get_diagF),
                                ('fric', self.get_diagFric), ('visc', self.get_diagVisc)):
            for prefix, data in zip(('te', 'tb', 'fe', 'fb'), method()):
                ds.variables[prefix+process+'k'][it] = data
        ds.variables['cfl'][it], ds.variables['damping_dt'][it] = self.get_cfl()
        ds.sync()

    def create_rst(self, nf, prefix='rst'):
        """Spectral restart including AB3 startup, forcing memory and RNG state."""
        nc = _netcdf()
        os.makedirs(self.savedir, exist_ok=True)
        path = os.path.join(self.savedir, f'{prefix}_{nf:04d}.nc')
        exists = os.path.exists(path)
        ds = nc.Dataset(path, 'a' if exists else 'w', format='NETCDF4', auto_complex=True)
        try:
            if exists:
                self._validate_file(ds)
                if (ds.ts_scheme != self.ts_scheme or ds.backend != self.backend
                        or ds.fft_backend != FFT_BACKEND):
                    raise ValueError('Restart append requires matching scheme and backend.')
            else:
                self._write_metadata(ds)
                for name, size in (('time', None), ('ind', 3), ('y', self.Ny), ('x', self.Nx)):
                    ds.createDimension(name, size)
                ds.createVariable('time', 'f8', ('time',)).units = 's'
                ds.createVariable('n_steps', 'i8', ('time',))
                ds.createVariable('history_count', 'i4', ('time',))
                dtype = 'c8' if self.precision == 'single' else 'c16'
                var = ds.createVariable('brst', dtype, ('time', 'ind', 'y', 'x'))
                var.description = 'Nondimensional full FFT: [older unforced RHS, previous unforced RHS, buoyancy].'
                for name in ('force_b', 'markov_hat'):
                    ds.createVariable(name, dtype, ('time', 'y', 'x')).units = 'nondimensional buoyancy tendency'
                ds.createVariable('rng_state', str, ('time',))
        except Exception:
            ds.close()
            raise
        self.rstds = ds

    def save_rst(self, it):
        ds = self.rstds
        ds.variables['time'][it] = self.t
        ds.variables['n_steps'][it] = self.n_steps
        ds.variables['history_count'][it] = self.history_count
        for ind, field in enumerate((self.k1_pp, self.k1_p, self.b_hat)):
            ds.variables['brst'][it, ind] = _host(field)
        ds.variables['force_b'][it] = _host(self.force_b)
        ds.variables['markov_hat'][it] = _host(self.markov_hat)
        ds.variables['rng_state'][it] = json.dumps(self._rng.bit_generator.state)
        ds.sync()

    def load_rst(self, filename, record=-1):
        nc = _netcdf()
        with nc.Dataset(filename, 'r', auto_complex=True) as ds:
            self._validate_file(ds)
            if ds.backend != self.backend or ds.fft_backend != FFT_BACKEND:
                raise ValueError('Use the saved FFT backend for reproducible restart.')
            history = np.asarray(ds.variables['brst'][record])
            if history.shape != (3, self.Ny, self.Nx) or not np.all(np.isfinite(history)):
                raise ValueError('Invalid spectral restart state.')
            n_steps = int(ds.variables['n_steps'][record])
            count = int(ds.variables['history_count'][record])
            time = float(ds.variables['time'][record])
            if n_steps < 0 or count not in (0, 1, 2) or not np.isclose(time, n_steps*self.dt, rtol=0, atol=1.e-8*self.dt):
                raise ValueError('Invalid restart time/history metadata.')
            rng = np.random.default_rng()
            rng.bit_generator.state = json.loads(ds.variables['rng_state'][record])
            self.k1_pp, self.k1_p, self.b_hat = [cp.asarray(a, dtype=self.cdtype) for a in history]
            self.force_b = cp.asarray(ds.variables['force_b'][record], dtype=self.cdtype)
            self.markov_hat = cp.asarray(ds.variables['markov_hat'][record], dtype=self.cdtype)
            self.ts_scheme = ds.ts_scheme
        self._rng = rng
        self.n_steps, self.history_count, self.t = n_steps, count, time
        self.is_not_rst = False
        self._update_state()

    # Plotting and visualization methods
    def plot_diag(self, save_path=None):
        import matplotlib.pyplot as plt
        fields = self.get_fields()
        te, _, fe, _ = self.get_diagNL()
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        im = axes[0, 0].imshow(_host(fields['b']), origin='lower', cmap='RdBu_r',
                                extent=(0, self.Lx/1000, 0, self.Ly/1000))
        axes[0, 0].set(title=f'Surface buoyancy, t={self.t/86400:.3f} days', xlabel='x [km]', ylabel='y [km]')
        fig.colorbar(im, ax=axes[0, 0], label='b [m s$^{-2}$]')
        axes[0, 1].loglog(self.kk_iso[1:], self.get_Ek()[1:])
        axes[0, 1].set(xlabel='k [rad m$^{-1}$]', ylabel='Column E shell sum [m$^3$ s$^{-2}$]')
        axes[1, 0].semilogx(self.kk_iso[1:], te[1:], label='Nonlinear')
        for label, method in (('Forcing', self.get_diagF), ('Drag', self.get_diagFric), ('Diffusion', self.get_diagVisc)):
            axes[1, 0].semilogx(self.kk_iso[1:], method()[0][1:], label=label)
        axes[1, 0].set(xlabel='k [rad m$^{-1}$]', ylabel='E transfer [m$^3$ s$^{-3}$]')
        axes[1, 0].legend()
        axes[1, 1].semilogx(self.kk_iso[1:], fe[1:])
        axes[1, 1].set(xlabel='k [rad m$^{-1}$]', ylabel='Nonlinear E flux [m$^3$ s$^{-3}$]')
        if save_path:
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
        else:
            plt.show()

    def save_snapshot(self, nstep):
        outdir = os.path.join(self.savedir, 'figs')
        os.makedirs(outdir, exist_ok=True)
        self.plot_diag(os.path.join(outdir, f'snap_{nstep:07d}.png'))

    # Main simulation loop
    def run(self, scheme=None, tmax=86400., tsave=200, tsave_rst=2000,
            nsave=100, savedir='run_0', saveplot=False):
        """Run to physical tmax [s]; save intervals count full timesteps.

        scheme=None retains the current scheme (initially AB3), including
        the scheme loaded from a restart. tmax must be a multiple of dt.
        Repeated calls continue the current
        state. New output files are numbered after existing files, avoiding
        overwrite. Fixed-step explicit RK4/AB3 require resolved advection and
        diffusion timescales; CFL and maximum damping*dt are saved for review.
        """
        scheme = self.ts_scheme if scheme is None else scheme
        if scheme not in ('ab3', 'rk4'):
            raise ValueError("scheme must be 'ab3' or 'rk4'.")
        if not np.isfinite(tmax) or tmax < self.t:
            raise ValueError('tmax must be finite and at least the current time.')
        target = int(round(tmax/self.dt))
        if not np.isclose(tmax/self.dt, target, rtol=0, atol=1.e-8):
            raise ValueError('tmax must be an integer multiple of dt.')
        for name, value in (('tsave', tsave), ('tsave_rst', tsave_rst), ('nsave', nsave)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if scheme != self.ts_scheme:
            self.history_count = 0
            self.k1_p.fill(0j)
            self.k1_pp.fill(0j)
        self.ts_scheme, self.savedir = scheme, os.fspath(savedir)
        os.makedirs(self.savedir, exist_ok=True)
        counters = {}
        for prefix in ('output', 'rst'):
            numbers = []
            for name in os.listdir(self.savedir):
                if name.startswith(prefix+'_') and name.endswith('.nc'):
                    suffix = name[len(prefix)+1:-3]
                    if suffix.isdecimal():
                        numbers.append(int(suffix))
            nf = max(numbers, default=-1)+1
            counters[prefix] = [nf, int(nsave)]
        try:
            while True:
                for prefix, interval, create, save, attr in (
                    ('output', tsave, self.create_nc, self.save_var, 'ds'),
                    ('rst', tsave_rst, self.create_rst, self.save_rst, 'rstds')):
                    if self.n_steps % interval == 0:
                        nf, record = counters[prefix]
                        if record == nsave:
                            old = getattr(self, attr, None)
                            if old is not None and old.isopen():
                                old.close()
                            create(nf)
                            nf, record = nf+1, 0
                        save(record)
                        counters[prefix] = [nf, record+1]
                        if prefix == 'output':
                            print(f'step {self.n_steps:7d} t={self.t:.3f} s E={self.get_Etot():.6e} U_rms={self.get_Vrms():.6e} m/s')
                            if saveplot:
                                self.save_snapshot(self.n_steps)
                if self.n_steps == target:
                    break
                self._step_forward()
                if not bool(_host(cp.all(cp.isfinite(self.b_hat)))):
                    raise FloatingPointError('Nonfinite SQG state; check dt, resolution and forcing amplitude.')
        finally:
            for attr in ('ds', 'rstds'):
                ds = getattr(self, attr, None)
                if ds is not None and ds.isopen():
                    ds.close()
        print('Done.')


# Familiar import name for scripts organized like turb2d.py; fields remain b.
QGModel = SQGModel
