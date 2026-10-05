
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from scipy.fft import fft2,rfft2,ifft2,fftshift,irfft2
import numpy_groupies as npg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import set_global_backend
set_global_backend(cufft)
import gc
import netCDF4 as nc
import os


class QGModel:
    """Nondimensional active-scalar model: q_hat = -(k²+gamma²)^(alpha/2) psi_hat.

    alpha=2 retains the Helmholtz/QG model; alpha=1, gamma=0 is SQG.
    Nonzero gamma with alpha!=2 defines a chosen screened extension, not
    the constant-stratification ocean model in turb_sqg.py. In that ocean
    convention the corresponding nondimensional active scalar is q=-b/(N0*U_ref).
    Legacy shell diagnostics are grid sums, truncated to kk_iso; divide by
    Nx*Ny for means. get_invariants() instead returns full-mode means.
    """
    def __init__(self, Nx, Ny, Lx=2*cp.pi, Ly=2*cp.pi, dt=0.001,
                 beta=0, gamma=0,
                 friction=0.01,k_friction = 20000,hyvisc = 0,hyperorder=1,sp_filtr=False,cl=0,
                 forcing=None,fscale=4,finput=None,famp=1.,precision='single',
                 alpha=2., forcing_norm=None):
        """Initialize QG model with grid and parameters

        Args:
            Nx, Ny: Grid dimensions
            Lx, Ly: Domain sizes
            dt: Time step
            beta: Background q gradient; planetary beta only in the QG case.
            gamma: Inverse screening length; inverse deformation radius at alpha=2.
            alpha: Finite nonzero inversion exponent (2=QG, 1 with gamma=0
                is SQG); negative values define generalized active-scalar models.
            friction: Friction coefficient acting on q
            k_friction: Wavenumber cutoff for friction
            hyperorder: Order of hyperviscosity on q (1=Laplacian, 2=Biharmonic, etc.)
            sp_filtr: Whether to apply spectral filter
            cl: Leith parameter for biharmonic viscosity
            forcing: Forcing type ('wind', 'thuburn', 'markov', 'kflow',
                'psi_kflow', None). psi_kflow prescribes the steady tendency
                Fpsi=famp*sin(fscale*y), converted to Fq by q_operator.
            fscale: Forcing wavenumber scale
            forcing_norm: None preserves legacy forcing. Explicit 'none'
                uses the raw q forcing; 'amplitude' fixes its spatial RMS
                to famp; 'enstrophy' fixes <q*Fq> to finput; 'energy' fixes
                -<psi*Fq> to finput; 'kinetic_energy' fixes the forcing
                contribution to d<|grad psi|^2/2>/dt to finput.
                These are full-grid spatial means.
                Injection targets hold at the RHS level; very small initial
                q can require a much smaller dt to resolve the feedback.
                CDA/CLE reuse the reference's forcing at each RHS stage;
                the injection target then applies to the reference only.
                Markov accepts only None/'none' and keeps its native process.
            finput: Nonnegative injection target for 'enstrophy'/'energy'/'kinetic_energy';
                required in those modes and unused otherwise. Enstrophy
                here means q variance/2; energy means -<psi*q>/2, including
                for SQG (where this energy differs from surface kinetic energy).
            famp: Spatial RMS target in 'amplitude' mode, or the native
                Markov amplitude. Legacy wind also uses famp; legacy kflow
                uses unit-amplitude velocity forcing f = sin(k_f*y)e_x.
                For psi_kflow, famp is the nonnegative peak amplitude of
                Fpsi; forcing_norm must be None or 'none' (no normalization).
            precision: 'single' (float32/complex64 state, the historical
                behavior) or 'double' (float64/complex128). Restart files are
                written at the working precision (c8/c16), so a run cannot
                resume from a restart of the other precision mid-file.
        """
        if precision not in ('single', 'double'):
            raise ValueError(f"precision must be 'single' or 'double', got {precision!r}")
        if not np.isfinite(alpha) or alpha == 0:
            raise ValueError('alpha must be finite and nonzero.')
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError('gamma must be finite and nonnegative.')
        self.alpha = float(alpha)
        if cl and self.alpha != 2:
            raise ValueError('Leith closure currently requires alpha=2.')
        if forcing not in (None, 'wind', 'thuburn', 'markov', 'kflow', 'psi_kflow'):
            raise ValueError(f'Unknown forcing scheme: {forcing!r}')
        if forcing_norm not in (None, 'none', 'amplitude', 'enstrophy', 'energy', 'kinetic_energy'):
            raise ValueError('forcing_norm must be None, none, amplitude, enstrophy, energy, or kinetic_energy.')
        if forcing in (None, 'markov') and forcing_norm not in (None, 'none'):
            raise ValueError('External and native Markov forcing bypass forcing normalization.')
        if forcing == 'psi_kflow' and forcing_norm not in (None, 'none'):
            raise ValueError("psi_kflow requires forcing_norm=None or 'none'; famp fixes Fpsi's peak amplitude.")
        if (forcing == 'psi_kflow' or forcing_norm == 'amplitude') and (not np.isfinite(famp) or famp < 0):
            raise ValueError('Forcing amplitude requires finite famp >= 0.')
        if forcing_norm in ('enstrophy', 'energy', 'kinetic_energy'):
            if finput is None or not np.isfinite(finput) or finput < 0:
                raise ValueError('Injection normalization requires an explicit finite finput >= 0.')
        if forcing == 'kflow' and self.alpha != 2 and forcing_norm is None:
            raise ValueError('Legacy kflow requires alpha=2; set forcing_norm explicitly to force q.')
        self.precision = precision
        self.rdtype = cp.float32 if precision == 'single' else cp.float64
        self.cdtype = cp.complex64 if precision == 'single' else cp.complex128
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = self.rdtype(Lx)
        self.Ly = self.rdtype(Ly)
        self.dt = self.rdtype(dt)
        self._init_grid()
        # QG parameters
        self.beta = beta # Coriolis gradient
        self.gamma = gamma # stratification parameter
        self.friction = friction # large scale friction
        self.k_friction = k_friction # wavenumber cutoff for friction
        # if hyperorder == 1:
        self.hyvisc = hyvisc
        # else:
        #     self.hyvisc = cp.float32(10**np.log2(2*hyperorder)/(self.Nx/2*(2*np.pi/self.Lx))**(2*hyperorder)) # viscosity t_eddy/kmax**(2n)
        self.hyperorder = hyperorder # order of hyper viscosity, 1-> Newnation 2-> biharmonic ...
        self.sp_filtr = sp_filtr # spectral filter impose on the tail of spectral (Arbic 2003)
        self.cl = cl #leith parameter
        self.forcing = forcing
        self.forcing_norm = forcing_norm
        self.fscale = fscale # scale of wind
        self.finput = finput # active only for explicit injection normalization
        self.famp = famp
        self.force_q = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self._raw_force_q = None
        self.da_term = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self.k1_p = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self.k1_pp = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
        self.is_not_rst = True

        # Default time-stepping scheme
        self.ts_scheme = 'ab3'
        self.t = self.rdtype(0.0)  # Initialize time
        self.n_steps = 0  # Initialize step counter
        
        self._prebuild_operator()
        self._my_div()
        # gpu or cpu?  backend
### private term
    def _init_grid(self):
        """Initialize computational grid and wavenumber arrays"""
        # FFT grids are periodic samples on [0, L); including both 0 and L
        # duplicates one point and leaks an integer-wavenumber forcing.
        self.x = cp.linspace(0, self.Lx, self.Nx, endpoint=False, dtype=self.rdtype)
        self.y = cp.linspace(0, self.Ly, self.Ny, endpoint=False, dtype=self.rdtype)
        self.x2d, self.y2d = cp.meshgrid(self.x,self.y)
        # Grid indices for FFT
        nx = cp.arange(self.Nx, dtype=self.rdtype); nx[int(self.Nx/2):] -= self.Nx # shift for proper wavenumbers
        ny = cp.arange(self.Ny, dtype=self.rdtype); ny[int(self.Ny/2):] -= self.Ny
        self.nx2d, self.ny2d = cp.meshgrid(nx,ny)
        # Wavenumbers
        kx = (2*cp.pi*nx/self.Lx).astype(self.rdtype)
        ky = (2*cp.pi*ny/self.Ly).astype(self.rdtype)
        # 2D wavenumber arrays
        self.kx2d, self.ky2d = cp.meshgrid(kx,ky)
        # 2D isotropic wavenumber magnitude
        self.kk = cp.sqrt(self.kx2d**2+self.ky2d**2).astype(self.rdtype)

        # Rounded |k| shell index.
        k_bins_grid = cp.round(cp.sqrt(self.nx2d**2 + self.ny2d**2)).astype('int')
        self.kk_idx_set, self.kk_idx = cp.unique(k_bins_grid, return_inverse=True)
        self.kk_set = self.kk_idx_set * (2*cp.pi/self.Lx)
        self.kk_range = self.kk_idx_set < int(self.Nx/2)
        self.kk_iso = self.kk_set[self.kk_range]
    def _init_friction(self):
        """Initialize friction mask for large-scale modes"""
        if self.friction:
            self.friction_mask = cp.zeros_like(self.kk)
            self.friction_mask[self.kk <= self.k_friction] = 1 # apply friction to large scales
        else:
            self.friction_mask=0

    def _init_filter(self):
        #"""Set up frictional filter (Arbic and Flierl, 2004)."""
        # 1. Define Grid Spacing (dx, dy)
        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny

        # 2. Calculate Dimensionless Wavenumber (k * dx)
        # This scales the wavenumber so Nyquist = pi
        wvx = cp.sqrt((self.kx2d * dx)**2 + (self.ky2d * dy)**2)

        # 3. Define Cutoff and Stiffness
        # Arbic uses 0.65 * Nyquist
        cphi = 0.65 * cp.pi 
        
        # Alpha (Stiffness). Arbic often uses 18.4 or 23.6.
        # You can make this a class parameter if you want.
        self.filterfac = 23.6 

        # 4. Compute the Filter
        # Initialize with ones (transparent)
        self.filtr = cp.ones_like(self.kx2d)
        
        # Mask for high wavenumbers
        mask = wvx > cphi
        
        # Apply Exponential Decay: exp( -alpha * (k*dx - cutoff)^4 )
        self.filtr[mask] = cp.exp(-self.filterfac * (wvx[mask] - cphi)**4)
        
        # Ensure the mean (0,0) is perfectly preserved (redundant but safe)
        self.filtr[0,0] = 1.0
        
    def _prebuild_operator(self):
        # Shared inversion symbol. Keep the alpha=2 arithmetic unchanged.
        k2 = self.kx2d**2+self.ky2d**2
        self.inversion_symbol = k2+self.gamma**2
        if self.alpha != 2:
            # Zero mean is excluded from inversion, also for negative powers.
            nonzero = self.inversion_symbol > 0
            self.inversion_symbol[nonzero] = self.inversion_symbol[nonzero]**(self.alpha/2)
        self.q_operator = -self.inversion_symbol
        self.inversion = cp.zeros_like(self.inversion_symbol)
        nonzero = self.inversion_symbol > 0
        self.inversion[nonzero] = -1/self.inversion_symbol[nonzero]
        self.inversion[0,0] = 0.0
        # laplacian
        self.lap = -(self.kx2d**2+self.ky2d**2)
        # hyperlap for hyperviscosity
        self.hylap = (-1)**(self.hyperorder+1)*self.hyvisc*(self.lap**(self.hyperorder))
        # filtr Arbic 2003
        if self.sp_filtr:
            self._init_filter()
        else:
            self.filtr = cp.ones_like(self.kx2d)

        if self.forcing == 'markov':
            self._init_markovforce(famp=self.famp)
        elif self.forcing == 'kflow':
            self._set_kflow_force()
        elif self.forcing == 'psi_kflow':
            self._set_psi_kflow_force()
        elif self.forcing == 'thuburn' and self.forcing_norm is not None:
            self._raw_force_q = fft2(0.1*cp.sin(32*np.pi*self.x2d))
        
        self._init_friction()
        self.linear_damping = -self.friction_mask*self.friction + self.hylap
        # preallocate for jacobian  for dealiasing
        self.Nxpad = int(3*self.Nx/2)
        self.Nypad = int(3*self.Ny/2)
        self.pad_buffer = cp.zeros((self.Nypad,self.Nxpad), dtype=self.cdtype)

        self.parseval_fac = (self.Nxpad*self.Nypad)/(self.Nx*self.Ny)
        # for truncating in padded field
        self.px0 = int((self.Nxpad-self.Nx)/2)
        self.px1= self.px0+self.Nx
        self.py0 = int((self.Nypad-self.Ny)/2)
        self.py1 = self.py0+self.Ny
        # Powers-of-two use this direct unshifted layout. Besides avoiding
        # fftshift-sized temporaries, it leaves the ambiguous even-grid
        # Nyquist row and column at zero.
        self._direct_padding = (
            self.Nx % 2 == 0 and self.Ny % 2 == 0
            and self.Nxpad % 2 == 0 and self.Nypad % 2 == 0
        )
        
        
    def _padding(self,ft):
        self.pad_buffer.fill(0j)
        if self._direct_padding:
            hx = self.Nx // 2
            hy = self.Ny // 2
            sx_neg = slice(hx + 1, self.Nx)
            sy_neg = slice(hy + 1, self.Ny)
            px_neg = slice(self.Nxpad - hx + 1, self.Nxpad)
            py_neg = slice(self.Nypad - hy + 1, self.Nypad)
            cp.multiply(ft[:hy, :hx], self.parseval_fac,
                        out=self.pad_buffer[:hy, :hx])
            cp.multiply(ft[:hy, sx_neg], self.parseval_fac,
                        out=self.pad_buffer[:hy, px_neg])
            cp.multiply(ft[sy_neg, :hx], self.parseval_fac,
                        out=self.pad_buffer[py_neg, :hx])
            cp.multiply(ft[sy_neg, sx_neg], self.parseval_fac,
                        out=self.pad_buffer[py_neg, px_neg])
            return self.pad_buffer

        #shift zero frequency to center
        self.pad_buffer[self.py0:self.py1,self.px0:self.px1]=fftshift(ft)
        if self.Nx % 2 == 0:
            self.pad_buffer[self.py0:self.py1, self.px0] = 0.0
        if self.Ny % 2 == 0:
            self.pad_buffer[self.py0, self.px0:self.px1] = 0.0
        # shift back than the low frequency will back to the edge 
        # garuntee the power is unchange to return the same value, i.e. parseval theorem
        self.pad_buffer *= self.parseval_fac
        return fftshift(self.pad_buffer)

    def _unpadding(self, ft_pad):
        if self._direct_padding:
            hx = self.Nx // 2
            hy = self.Ny // 2
            sx_neg = slice(hx + 1, self.Nx)
            sy_neg = slice(hy + 1, self.Ny)
            px_neg = slice(self.Nxpad - hx + 1, self.Nxpad)
            py_neg = slice(self.Nypad - hy + 1, self.Nypad)
            ft = cp.zeros((self.Ny, self.Nx), dtype=ft_pad.dtype)
            inv_fac = 1.0 / self.parseval_fac
            cp.multiply(ft_pad[:hy, :hx], inv_fac, out=ft[:hy, :hx])
            cp.multiply(ft_pad[:hy, px_neg], inv_fac, out=ft[:hy, sx_neg])
            cp.multiply(ft_pad[py_neg, :hx], inv_fac, out=ft[sy_neg, :hx])
            cp.multiply(ft_pad[py_neg, px_neg], inv_fac, out=ft[sy_neg, sx_neg])
            return ft

        ft_shift = fftshift(ft_pad)[self.py0:self.py1,self.px0:self.px1]
        ft = fftshift(ft_shift/self.parseval_fac)
        return self._enforce_spectral_constraints(ft)

    def _enforce_spectral_constraints(self, ft):
        """Project a full FFT field onto the unambiguous real-field subspace."""
        if self.Nx % 2 == 0:
            ft[:, self.Nx // 2] = 0.0
        if self.Ny % 2 == 0:
            ft[self.Ny // 2, :] = 0.0
        ft[0, 0] = ft[0, 0].real
        return ft

        
    def _get_rhs(self,q_hat,time=None,force_q=None,forcing_out=None):
        """Compute right-hand side of QG dynamics equation
        
        Returns the tendency from all processes: advection, beta effect, damping
        """
        p_hat = self.inversion*q_hat # invert to get streamfunction
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        damping_term = self.linear_damping*q_hat
        if self.cl:
            damping_term += self._compute_leith_term(q_hat)

        rhs = damping_term - jacobian_term
        if self.beta:
            rhs -= self.beta*self.kx2d*1j*p_hat
        if force_q is None:
            force_q = self._forcing_at_state(q_hat, p_hat, time)
        if forcing_out is not None:
            forcing_out.append(force_q.copy())
        rhs += force_q
        rhs += self.da_term
        return self._enforce_spectral_constraints(rhs)

    def _relative_vorticity(self, p_hat, q_hat):
        """Diagnose Delta psi, retaining legacy alpha=2 arithmetic."""
        if self.alpha == 2:
            return q_hat + self.gamma**2*p_hat if self.gamma else q_hat
        return self.lap*p_hat

    def _q_from_psi(self, p_hat):
        if self.alpha == 2:
            return self.lap*p_hat - self.gamma**2*p_hat
        return self.q_operator*p_hat

    def _rk4(self, q_hat, forcing_stages=None, forcing_out=None):
        """Compute 4th order Runge-Kutta stages"""
        forces = (None,) * 4 if forcing_stages is None else forcing_stages
        k1 = self._get_rhs(q_hat, self.t, forces[0], forcing_out)
        k2 = self._get_rhs(q_hat + 0.5 * self.dt * k1, self.t + 0.5*self.dt,
                           forces[1], forcing_out)
        k3 = self._get_rhs(q_hat + 0.5 * self.dt * k2, self.t + 0.5*self.dt,
                           forces[2], forcing_out)
        k4 = self._get_rhs(q_hat + self.dt * k3, self.t + self.dt,
                           forces[3], forcing_out)

        return k1,k2,k3,k4

    def _step_forward(self, forcing_stages=None, forcing_out=None):
        """Advance one step, optionally recording or accepting reference forcing.

        A forcing list contains the one AB3 or four RK4 RHS values, followed
        by the endpoint value for diagnostics. Default stepping is unchanged.
        """
        if forcing_out is not None:
            forcing_out.clear()
        if forcing_stages is not None:
            rk4_step = self.ts_scheme == 'rk4' or (self.is_not_rst and self.n_steps < 2)
            if len(forcing_stages) != (5 if rk4_step else 2):
                raise ValueError('Reference and replica must use matching RHS stages.')
        elif self.forcing == 'wind' and self.forcing_norm is None:
            self._set_windforce()
        elif self.forcing =='thuburn' and self.forcing_norm is None:
            self.force_q = fft2(0.1*cp.sin(32*np.pi*self.x2d))
        elif self.forcing == 'markov':
            self._update_markovforce()
        # kflow and psi_kflow are steady and were built during __init__.
        self._enforce_spectral_constraints(self.q_hat)
        q = self.q_hat

        if self.ts_scheme=='rk4':
            # 4th order Runge-Kutta integration
            k1,k2,k3,k4 = self._rk4(q, forcing_stages, forcing_out)
            self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        elif self.ts_scheme == 'ab3':
            # 3rd order Adams-Bashforth integration
            if self.is_not_rst:
                if self.n_steps == 0:
                    k1,k2,k3,k4 = self._rk4(q, forcing_stages, forcing_out)
                    self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    self.k1_pp = k1 # store RHS history for AB3
                elif self.n_steps == 1:
                    k1,k2,k3,k4 = self._rk4(q, forcing_stages, forcing_out)
                    self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    self.k1_p = k1
                else:
                    k1 = self._get_rhs(q, force_q=None if forcing_stages is None else forcing_stages[0],
                                       forcing_out=forcing_out)
                    self.q_hat = q + self.dt/12*(23*k1-16*self.k1_p+5*self.k1_pp)
                    self.k1_pp = self.k1_p
                    self.k1_p = k1
            else:
                k1 = self._get_rhs(q, force_q=None if forcing_stages is None else forcing_stages[0],
                                   forcing_out=forcing_out)
                self.q_hat = q + self.dt/12*(23*k1-16*self.k1_p+5*self.k1_pp)
                self.k1_pp = self.k1_p
                self.k1_p = k1

        self.q_hat *=self.filtr # apply spectral filter
        self._enforce_spectral_constraints(self.q_hat)
        self.p_hat = self.inversion*self.q_hat
        self.rv_hat = self._relative_vorticity(self.p_hat, self.q_hat)
        self.force_q = (self._forcing_at_state(self.q_hat, self.p_hat, self.t + self.dt)
                        if forcing_stages is None else forcing_stages[-1].copy())
        if forcing_out is not None:
            forcing_out.append(self.force_q.copy())
        
    
    def _compute_jacobian(self,p_hat,q_hat):
        """Compute Jacobian term for non-linear advection with dealiasing
        
        Uses 3/2-rule padding to avoid aliasing errors in spectral computation
        """
        dxp_hat = self.kx2d*1j*p_hat
        dyp_hat = self.ky2d*1j*p_hat
        dxq_hat = self.kx2d*1j*q_hat
        dyq_hat = self.ky2d*1j*q_hat
        # Zero-padding with 3/2 rule to remove aliasing
        dxq_r = ifft2(self._padding(dxq_hat)).real
        dyq_r = ifft2(self._padding(dyq_hat)).real
        dxp_r = ifft2(self._padding(dxp_hat)).real
        dyp_r = ifft2(self._padding(dyp_hat)).real
    
        jacob_r = dyq_r*dxp_r-dyp_r*dxq_r
    
        jacob_hat = self._unpadding(fft2(jacob_r))
    
        return jacob_hat

    def _compute_leith_term(self, q_hat=None):
        """Compute biharmonic Leith viscosity term
        
        Args:
            q_hat: Active scalar in Fourier space (uses current value if None)
        """
        if self.cl == 0:
            return 0.0
        if q_hat is None:
            q_hat = self.q_hat # use current value for stable stepping
        dxq_hat = self.kx2d*1j*q_hat
        dyq_hat = self.ky2d*1j*q_hat
        
        dxq_r = ifft2(self._padding(dxq_hat)).real
        dyq_r = ifft2(self._padding(dyq_hat)).real
        
        grad_q_r = cp.sqrt(dxq_r**2+dyq_r**2)
        
        nabla = self.Lx/self.Nx
        nu_e = (self.cl*nabla)**3*grad_q_r # eddy viscosity
        
        flux_x_r = nu_e*dxq_r
        flux_y_r = nu_e*dyq_r

        flux_x_hat = self.kx2d*1j*self._unpadding(fft2(flux_x_r))
        flux_y_hat = self.ky2d*1j*self._unpadding(fft2(flux_y_r))

        return flux_x_hat+flux_y_hat

    #TODO self spec_pad
    

    def spec_cut(self,phi_hat):
        hNx = phi_hat.shape[0] 
        hNy = phi_hat.shape[1]
        hx0 = int((hNx-self.Nx)/2)
        hx1 = hx0+self.Nx
        hy0 = int((hNy-self.Ny)/2)
        hy1 = hy0+self.Ny
        phi_cut = fftshift(fftshift(phi_hat)[hx0:hx1,hy0:hy1])
        return phi_cut        


### initial term
    def _norm_energy(self):
            self._enforce_spectral_constraints(self.p_hat)
            self.rv_hat = self.lap*self.p_hat
            # initial potential vorticity (q)
            self.q_hat = self._q_from_psi(self.p_hat)
            # normalize mean of total energy to 0.5
            ene_tot = self.get_Etot(self.p_hat)
            norm_fac = cp.sqrt(self.eini/(ene_tot/(self.Nx*self.Ny)))
            self.p_hat = norm_fac*self.p_hat
            # initial relavitve vorticity (rv)
            self.rv_hat = self.lap*self.p_hat
            # initial potential vorticity (q)
            self.q_hat = self._q_from_psi(self.p_hat)
        
    def set_initial_condition(self,scheme='gauss',eini=0,q_ini=None,trst=0):
        """Initialize nondimensional q. 'field' accepts a real (Ny,Nx) field
        or its Hermitian full FFT. eini is mean generalized energy.
        Raw 'rst' arrays must come from the same alpha/gamma/damping model;
        unlike a NetCDF append, an array contains no model metadata to check.
        """
        self.eini = eini
        self.trst = trst
        if scheme == 'field':
            field = cp.asarray(q_ini)
            if field.shape != (self.Ny, self.Nx) or not bool(cp.all(cp.isfinite(field))):
                raise ValueError('q_ini must be a finite (Ny,Nx) field or full FFT.')
            if field.dtype.kind == 'c':
                physical = ifft2(field)
                scale = max(float(cp.max(cp.abs(physical))), np.finfo(float).tiny)
                if float(cp.max(cp.abs(physical.imag))) > 100*np.finfo(self.rdtype).eps*scale:
                    raise ValueError('Spectral q_ini must represent a real field.')
                self.q_hat = field.astype(self.cdtype).copy()
            else:
                self.q_hat = fft2(field.astype(self.rdtype))
            self._enforce_spectral_constraints(self.q_hat)
            self.p_hat = self.inversion*self.q_hat
            if eini:
                self._norm_energy()
        elif scheme == 'jcm1984':
            k_peak = 6
            # kk**(-A)*(1 + (kk/k0)**4)**(-B)
            # generate Fourier conponent of initial streamfunction field
            ls=3
            ss=-3
            A = 3-ls
            B = (3-ss-A)/4
            amp = cp.sqrt(self.kk**(-A)*(1 + (self.kk/self.k_peak)**4)**(-B))
            rand_p = fft2(cp.random.randn(*self.kk.shape).astype(self.rdtype))
            rand_p = rand_p*amp
            rand_p[0,0] = 0.0
            self.p_hat = rand_p.copy()
            self._norm_energy()
        elif scheme =='thuburn':
            # only valid for unit domain
            q_ini = cp.sin(8*cp.pi*self.x2d)*cp.sin(8*cp.pi*self.y2d)+ \
                     0.4*cp.cos(6*cp.pi*self.x2d)*cp.cos(6*cp.pi*self.y2d)+ \
                     0.3*cp.cos(10*cp.pi*self.x2d)*cp.cos(10*cp.pi*self.y2d)+\
                     0.02*cp.sin(2*cp.pi*self.x2d)+0.02*cp.sin(2*cp.pi*self.y2d)
            fq_ini = fft2(q_ini)
            self.q_hat = fq_ini.copy()
            self.p_hat = self.inversion*self.q_hat
            self.rv_hat= self.lap*self.p_hat
            # self._norm_energy()

        elif scheme == 'gauss':
            psi_phys = cp.random.randn(self.Ny, self.Nx).astype(self.rdtype)
            rand_p = fft2(psi_phys)
            fmask = cp.zeros_like(self.kk)
        
            # Select wavenumbers inside the shell
            idx = (self.kk >= 3) & (self.kk <= 5)
            fmask[idx] = 1.0
            
            # Ensure the mean (k=0) is never forced
            fmask[0, 0] = 0.0

            # fix norm
            n_modes = cp.sum(fmask)

            norm_fac = 1/ cp.sqrt(n_modes)
            self.p_hat = rand_p.copy()*fmask
            
            if eini:
                self._norm_energy()
                
            self.rv_hat = self.lap * self.p_hat
            self.q_hat = self._q_from_psi(self.p_hat)
        elif scheme == 'kflow':
            # Kolmogorov-flow CDA starts from q_tilde(0)=0.
            # The observed low modes p(0) are inserted by cda_turb2d.py:ot2003.
            self.q_hat = cp.zeros((self.Ny, self.Nx), dtype=self.cdtype)
            self.p_hat = cp.zeros_like(self.q_hat)
            self.rv_hat = cp.zeros_like(self.q_hat)
        elif scheme == 'rst':
            # Restart from a saved state. The state may be supplied either as
            # spectral coefficients (complex -> the new bit-exact format) or as
            # a physical field (real -> legacy format, lifted via an FFT).
            q_ini = cp.asarray(q_ini)
            is_spectral = q_ini.dtype.kind == 'c'

            def _to_qhat(field):
                if is_spectral:
                    return field.astype(self.cdtype)
                return fft2(field.astype(self.rdtype))

            if q_ini.ndim == 2:
                self.q_hat = _to_qhat(q_ini.copy())
                self.p_hat = self.inversion*self.q_hat
                self.rv_hat= self.lap*self.p_hat
            elif q_ini.ndim == 3 :
                self.is_not_rst = False
                self.q_hat = _to_qhat(q_ini[2,:,:].squeeze())
                self.p_hat = self.inversion*self.q_hat
                self.rv_hat= self.lap*self.p_hat
                self.k1_p = _to_qhat(q_ini[1,:,:].squeeze())
                self.k1_pp  = _to_qhat(q_ini[0,:,:].squeeze())
            if eini:
                self._norm_energy()

        elif scheme == 'fromhr':
            fq_ini = fft2(cp.array(q_ini))
            hNx = q_ini.shape[0]
            hNy = q_ini.shape[1]
            hx0 = int((hNx-self.Nx)/2)
            hx1 = hx0+self.Nx
            hy0 = int((hNy-self.Ny)/2)
            hy1 = hy0+self.Ny
            self.q_hat = fftshift(fftshift(fq_ini)[hx0:hx1,hy0:hy1])
            norm_fac = (self.Nx*self.Ny)/(hNx*hNy)
            self.q_hat *=norm_fac
            self.p_hat = self.inversion*self.q_hat
            self.rv_hat = self.lap*self.p_hat
        else:
            raise ValueError(f'Unknown initial-condition scheme: {scheme!r}')
        self._enforce_spectral_constraints(self.q_hat)
        self._enforce_spectral_constraints(self.k1_p)
        self._enforce_spectral_constraints(self.k1_pp)
        self.p_hat = self.inversion*self.q_hat
        self.rv_hat = self._relative_vorticity(self.p_hat, self.q_hat)
        self.Etot = self.get_Etot(self.p_hat)
        self.Ek = self.get_Ek(self.p_hat)
        self.tenlk, self.tqnlk, self.fenlk, self.fqnlk = self.get_diagNL(self.p_hat,self.q_hat)
        self.force_q = self._forcing_at_state(self.q_hat, self.p_hat, self.trst)
     
### forcing term 
    def _forcing_at_state(self, q_hat, p_hat=None, time=None):
        """Evaluate deterministic forcing at this RHS state and time.

        Native Markov and legacy/external forcing are held fixed through
        each step. Explicit modes project the raw pattern before measuring
        its RMS or work, using Parseval over the full FFT (not shell sums).
        Injection control multiplies by a signed scalar: a negative raw
        work reverses the pattern, and zero/near-zero work raises an error.
        """
        if self.forcing_norm is None or self.forcing in (None, 'markov'):
            return self.force_q
        if self.forcing == 'wind':
            time = self.t if time is None else time
            phi_x = cp.pi*cp.sin(1.5*time)
            phi_y = cp.pi*cp.sin(1.4*time)
            raw = fft2(cp.cos(self.fscale*self.y2d + phi_y)
                       - cp.cos(self.fscale*self.x2d + phi_x))
        else:
            raw = self._raw_force_q.copy()
        self._enforce_spectral_constraints(raw)
        raw[0, 0] = 0.0
        if self.forcing_norm == 'none':
            return raw.astype(self.cdtype)
        target = self.famp if self.forcing_norm == 'amplitude' else self.finput
        if target == 0:
            return cp.zeros_like(raw, dtype=self.cdtype)
        mean_fac = 1.0/(self.Nx*self.Ny)**2
        raw2 = float(cp.sum(cp.abs(raw)**2, dtype=cp.float64))*mean_fac
        if not np.isfinite(raw2) or raw2 <= 0:
            raise ValueError('Cannot normalize a zero or nonfinite forcing pattern.')
        if self.forcing_norm == 'amplitude':
            scale = target/np.sqrt(raw2)
        else:
            if p_hat is None:
                p_hat = self.inversion*q_hat
            gradient = q_hat if self.forcing_norm == 'enstrophy' else -p_hat
            if self.forcing_norm == 'kinetic_energy':
                # dK/dq = k^2 * inversion * psi, with K=<|grad psi|^2>/2.
                gradient = self.kk**2*self.inversion*p_hat
            work = float(cp.sum(cp.real(cp.conj(gradient)*raw), dtype=cp.float64))*mean_fac
            work_scale = float(cp.sum(cp.abs(gradient)*cp.abs(raw), dtype=cp.float64))*mean_fac
            tolerance = 32*np.finfo(self.rdtype).eps*work_scale
            if not np.isfinite(work) or not np.isfinite(tolerance) or abs(work) <= tolerance:
                raise ValueError('Cannot impose finput: forcing work is zero or near zero; '
                                 'use a nonzero initial field overlapping the forcing pattern.')
            scale = target/work
        force = (scale*raw).astype(self.cdtype)
        if not bool(cp.all(cp.isfinite(force))):
            raise ValueError('Normalized forcing exceeds the working precision; reduce the target.')
        return force

    def _set_kflow_force(self):
        # Kolmogorov velocity forcing
        # f = sin(k_f y) e_x. The vorticity equation receives curl(f).
        # Explicit modes use this same q pattern for any inversion exponent.
        Fq = -self.fscale*cp.cos(self.fscale*self.y2d)
        Fq = Fq.astype(self.rdtype)
        Fq -= cp.mean(Fq)
        self.force_q = fft2(Fq)
        if self.forcing_norm is not None:
            self._raw_force_q = self.force_q.copy()

    def _set_psi_kflow_force(self):
        """Prescribe a periodic single-mode Fpsi, using the model's q sign."""
        if not np.isfinite(self.fscale) or self.fscale <= 0:
            raise ValueError('psi_kflow requires finite fscale > 0.')
        mode = self.fscale*float(self.Ly)/(2*np.pi)
        mode_index = int(round(mode))
        tolerance = 32*np.finfo(self.rdtype).eps*max(1.0, abs(mode))
        if abs(mode-mode_index) > tolerance or mode_index < 1:
            raise ValueError('psi_kflow requires fscale*Ly/(2*pi) to be a positive integer.')
        if mode_index >= (self.Ny+1)//2:
            raise ValueError('psi_kflow forcing must lie below the Nyquist cutoff.')
        Fpsi_hat = fft2((self.famp*cp.sin(self.fscale*self.y2d)).astype(self.rdtype))
        self._enforce_spectral_constraints(Fpsi_hat)
        Fpsi_hat[0, 0] = 0.0
        self.force_q = (self.q_operator*Fpsi_hat).astype(self.cdtype)
        if not bool(cp.all(cp.isfinite(self.force_q))):
            raise ValueError('psi_kflow forcing exceeds the working precision.')
        self._raw_force_q = self.force_q.copy()

    def _set_windforce(self):
        # graham 2013 and Frezat 2022
        # only valid for 2pi domain
        phi_x = cp.pi*cp.sin(1.5*self.t)
        phi_y = cp.pi*cp.sin(1.4*self.t)
        Fq = cp.cos(self.fscale*self.y2d + phi_y) - cp.cos(self.fscale*self.x2d + phi_x)  # original frezat& graham
        # Fq = cp.sin(self.fscale*self.y2d )  # horizontal shear
        normF = cp.linalg.norm(Fq) /self.Nx # fix  L2 norm

        norm_fac = self.famp/normF

        Fq_hat = fft2(norm_fac*Fq) # amplified to get large energy
        
        self.force_q = Fq_hat
        
        # plt.imshow(ifft2(Fq_hat).real.get())
        # plt.colorbar()
        # plt.show()
    
    def _init_markovforce(self, famp=1.0, t_r=0.5):
        # markovian forcing 
        # from Maltrud and Vallis 1991
        k_min = self.fscale-2
        k_max = self.fscale+2
        
        #  Create the Spectral Mask
        #  forcing only to a specific wavenumber shell.
        self.fmask = cp.zeros_like(self.kk)
        
        # Select wavenumbers inside the shell
        idx = (self.kk >= k_min) & (self.kk <= k_max)
        self.fmask[idx] = 1.0
        
        # Ensure the mean (k=0) is never forced
        self.fmask[0, 0] = 0.0

        # fix norm
        n_modes = cp.sum(self.fmask)

        norm_fac = 1/ cp.sqrt(n_modes)

        # Markov Coefficient R 
        # R depends on the timestep dt and correlation time t_r
        # If t_r = 0, R = 0 (White Noise). 
        if t_r == 0:
            self.fR = self.rdtype(0.0)
        else:
            self.fR = cp.exp(-self.dt / t_r).astype(self.rdtype)
        self.fseed = 10
        #  Pre-calculate the amplitude coefficient: A * sqrt(1 - R^2)
        # This ensures the variance of the forcing stays constant at A^2 over time.
        self.fcoef = (norm_fac*famp * cp.sqrt(1 - self.fR**2)).astype(self.rdtype)

        # Initialize the forcing field F_{n-1} to zero
        self.force_q = cp.zeros((self.Nx, self.Ny), dtype=self.cdtype)

    def _update_markovforce(self):
        """Update Markovian stochastic forcing with temporal correlation
        
        Uses correlated noise to maintain consistent forcing amplitude
        """
        # Generate random noise
        cp.random.seed(self.fseed)
        if self.fseed < 42:
            self.fseed +=1
        else:
            self.fseed = 10
        noise_phys = cp.random.randn(self.Nx, self.Ny).astype(self.rdtype)
        noise_hat = fft2(noise_phys)
        
        # Apply spectral mask to specific wavenumber shell
        noise_hat *= self.fmask
        
        # Markov update with temporal correlation (F_n = a*noise + r*F_{n-1})
        self.force_q = (self.fcoef * noise_hat) + (self.fR * self.force_q)

## diagnostic term
    def get_Ek(self,p_hat):
        """Generalized energy -<psi q>/2, in legacy grid-sum shell units.
        
        Integrates energy density in spectral shells using Parseval's theorem
        """
        # Energy density in spectral space
        ene_dens = 0.5*self.inversion_symbol*cp.abs(p_hat)**2
        # Physical space using Parseval's Theorem
        norm_fac = 1/(self.Nx*self.Ny)
        ene_kk = npg.aggregate(self.kk_idx.ravel().get(),ene_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ene_kk
    def get_Etot(self,p_hat):
        """Generalized energy summed over retained isotropic shells, in grid-sum units."""
        ene_kk = self.get_Ek(p_hat)
        ene_tot = np.sum(ene_kk) 
        return ene_tot

    def get_Vrms(self,p_hat):
        """RMS velocity from the retained isotropic shells."""
        ene_tot = self.get_Ktot(p_hat)
        vrms = np.sqrt(2*ene_tot/(self.Nx*self.Ny))
        return vrms

    def get_Kk(self, p_hat):
        """Kinetic energy, independent of alpha/gamma; legacy shell units."""
        density = 0.5*self.kk**2*cp.abs(p_hat)**2
        return npg.aggregate(self.kk_idx.ravel().get(), density.ravel().get(),
                             func='sum')[self.kk_range.get()] / (self.Nx*self.Ny)

    def get_Ktot(self, p_hat):
        """Kinetic energy summed over retained isotropic shells, in grid-sum units."""
        return float(np.sum(self.get_Kk(p_hat)))

    def get_invariants(self, q_hat=None):
        """Full retained-mode horizontal means (no isotropic-shell cutoff).

        E=-<psi q>/2 and Q=<q²>/2 are inviscid invariants. K=<|grad psi|²>/2
        equals E for alpha=2,gamma=0 and Q for alpha=1,gamma=0, at zero mean.
        Z=<omega²>/2 is relative-vorticity enstrophy, generally not an invariant.
        get_Etot/get_Qtot/get_Ztot/get_Ktot instead return truncated shell grid sums;
        divide those by Nx*Ny for spatial means over the retained shells.
        """
        q_hat = self.q_hat if q_hat is None else q_hat
        p_hat = self.inversion*q_hat
        norm = (self.Nx*self.Ny)**2
        return dict(E=float(cp.sum(0.5*self.inversion_symbol*cp.abs(p_hat)**2))/norm,
                    Q=float(cp.sum(0.5*cp.abs(q_hat)**2))/norm,
                    Z=float(cp.sum(0.5*cp.abs(self._relative_vorticity(p_hat, q_hat))**2))/norm,
                    K=float(cp.sum(0.5*self.kk**2*cp.abs(p_hat)**2))/norm)
    def get_Qrms(self,q_hat):
        """RMS active scalar from the retained isotropic shells."""
        ens_tot = self.get_Qtot(q_hat)
        qrms = np.sqrt(2*ens_tot/(self.Nx*self.Ny))
        return qrms
    def get_Qk(self,q_hat):
        """Q spectrum (half active-scalar mean square), in grid-sum units.

        This is enstrophy for ordinary 2D turbulence and half the active-scalar
        variance for zero-mean q. Only retained isotropic shells are returned.
        """
        # Half scalar mean-square density in spectral space
        ens_dens = 0.5*np.abs(q_hat)**2
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic scalar mean-square spectrum using Parseval's theorem
        ens_kk = npg.aggregate(self.kk_idx.ravel().get(),ens_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ens_kk
    def get_Qtot(self,q_hat):
        """Half active-scalar mean square summed over retained isotropic shells, in grid-sum units."""
        ens_kk = self.get_Qk(q_hat)
        ens_tot = np.sum(ens_kk)
        return ens_tot

    def get_Zk(self, rv_hat):
        """Relative-vorticity enstrophy spectrum, in grid-sum shell units."""
        return self.get_Qk(rv_hat)

    def get_Ztot(self, rv_hat):
        """Relative-vorticity enstrophy over retained isotropic shells, in grid-sum units."""
        return float(np.sum(self.get_Zk(rv_hat)))

    def get_sigma(self, z_kk=None):
        """Cumulative deformation rate sqrt(sum_{j<=k} Z(j)).

        Zk uses grid-sum shell units, so divide by Nx*Ny for spatial means
        before taking the square root. This follows the retained rounded shells.
        """
        z_kk = self.get_Zk(self.rv_hat) if z_kk is None else z_kk
        return np.sqrt(np.cumsum(z_kk)/(self.Nx*self.Ny))

    def get_TENL(self,p_hat,q_hat):
        """Compute spectral energy transfer from non-linear advection"""
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # Generalized-energy transfer: T_E = Re(p* * J), since dq/dt = -J.
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        return tenl
    
    def get_TQNL(self,p_hat,q_hat):
        """Compute spectral scalar-variance transfer from non-linear advection"""
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # Scalar-variance transfer: T_Q = -Re(q* * J)
        tqnl = -cp.real(cp.conj(q_hat)*jacobian_term)
        return tqnl

    def get_diagNL(self,p_hat,q_hat):
        """Compute generalized-energy and scalar-variance budgets from non-linear advection
        
        Returns spectral transfer and flux for both quantities
        """
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # spectral energy transfer of non-linear advection
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        # spectral scalar-variance transfer of non-linear advection
        tqnl = -cp.real(cp.conj(q_hat)*jacobian_term)

        # Isotropic spectrum of energy transfer of non-linear advection 
        # In physical space using spectral aggregation
        norm_fac = 1/(self.Nx*self.Ny)
        tenl_kk = npg.aggregate(self.kk_idx.ravel().get(),tenl.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer of non-linear advection
        # In physical space 
        tqnl_kk = npg.aggregate(self.kk_idx.ravel().get(),tqnl.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of non-linear advection
        fenl_kk = -np.cumsum(tenl_kk)
        # Isotropic spectrum of scalar variance flux of non-linear advection
        fqnl_kk = -np.cumsum(tqnl_kk)

        return tenl_kk, tqnl_kk, fenl_kk, fqnl_kk

    def get_diagF(self,p_hat,q_hat,force_q):
        """Compute generalized-energy and scalar-variance budgets from forcing
        
        Returns spectral transfer and flux for both quantities
        """
        # spectral energy transfer of forcing
        teF = -cp.real(cp.conj(p_hat)*force_q)
        # spectral scalar-variance transfer of forcing
        tqF = cp.real(cp.conj(q_hat)*force_q)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer 
        # In physical space using spectral aggregation
        teF_kk = npg.aggregate(self.kk_idx.ravel().get(),teF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer
        # In physical space 
        tqF_kk =npg.aggregate(self.kk_idx.ravel().get(),tqF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of forcing
        feF_kk = np.cumsum(teF_kk[::-1])[::-1]
        # Isotropic spectrum of scalar variance flux of forcing
        fqF_kk = np.cumsum(tqF_kk[::-1])[::-1]

        return teF_kk, tqF_kk, feF_kk, fqF_kk

    def get_diagDa(self,p_hat,q_hat,da_term):
        """Compute generalized-energy and scalar-variance budgets from data assimilation nudging
        
        Returns spectral transfer and flux for both quantities
        """
        # spectral energy transfer of DA
        teda = -cp.real(cp.conj(p_hat)*da_term)
        # spectral scalar-variance transfer of DA
        tqda = cp.real(cp.conj(q_hat)*da_term)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer 
        # In physical space using spectral aggregation
        teda_kk = npg.aggregate(self.kk_idx.ravel().get(),teda.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer
        # In physical space 
        tqda_kk =npg.aggregate(self.kk_idx.ravel().get(),tqda.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux
        feda_kk = np.cumsum(teda_kk[::-1])[::-1]
        # Isotropic spectrum of scalar variance flux
        fqda_kk = np.cumsum(tqda_kk[::-1])[::-1]

        return teda_kk, tqda_kk, feda_kk, fqda_kk

    def get_diagFric(self,p_hat,q_hat):
        """Compute generalized-energy and scalar-variance dissipation from large-scale friction
        
        Returns spectral dissipation and flux for both quantities
        """
        # spectral energy transfer of friction
        tendency = -self.friction_mask*self.friction*q_hat
        tefric = -cp.real(cp.conj(p_hat)*tendency)
        tqfric = cp.real(cp.conj(q_hat)*tendency)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of friction
        # In physical space using spectral aggregation
        tefric_kk = npg.aggregate(self.kk_idx.ravel().get(),tefric.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer of friction
        # In physical space 
        tqfric_kk = npg.aggregate(self.kk_idx.ravel().get(),tqfric.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of friction
        fefric_kk = np.cumsum(tefric_kk[::-1])[::-1]
        # Isotropic spectrum of scalar variance flux of friction
        fqfric_kk = np.cumsum(tqfric_kk[::-1])[::-1]
        return tefric_kk, tqfric_kk, fefric_kk, fqfric_kk

    def get_diagVisc(self, p_hat, q_hat):
        """Compute generalized-energy and scalar-variance dissipation from hyperviscosity
        
        Returns spectral dissipation and flux for both quantities
        """
        tendency = self.hylap*q_hat
        if self.cl:
            tendency = tendency + self._compute_leith_term(q_hat)
        # spectral energy transfer of viscosity
        tevisc = -cp.real(cp.conj(p_hat)*tendency)
        # spectral scalar-variance transfer of viscosity
        tqvisc = cp.real(cp.conj(q_hat)*tendency)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of viscosity
        # In physical space using spectral aggregation
        tevisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tevisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer of viscosity
        # In physical space 
        tqvisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tqvisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of viscosity
        fevisc_kk = np.cumsum(tevisc_kk[::-1])[::-1]
        # Isotropic spectrum of scalar variance flux of viscosity
        fqvisc_kk = np.cumsum(tqvisc_kk[::-1])[::-1]
        return tevisc_kk, tqvisc_kk, fevisc_kk, fqvisc_kk

    def get_diagFilt(self,p_hat,q_hat):
        """Compute energy and enstrophy loss from spectral filter
        
        Returns spectral dissipation and flux for both quantities
        """
        filt_rate = (self.filtr - 1.) / self.dt
        # spectral energy transfer of filter
        tefilt = -cp.real(cp.conj(p_hat) * filt_rate*q_hat)
        # spectral scalar-variance transfer of filter
        tqfilt = cp.real(cp.conj(q_hat) * filt_rate*q_hat)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of filter
        # In physical space using spectral aggregation
        tefilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tefilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of scalar variance transfer of filter
        # In physical space 
        tqfilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tqfilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of filter
        fefilt_kk = np.cumsum(tefilt_kk[::-1])[::-1]
        # Isotropic spectrum of scalar variance flux of filter
        fqfilt_kk = np.cumsum(tqfilt_kk[::-1])[::-1]
        return tefilt_kk, tqfilt_kk, fefilt_kk, fqfilt_kk
    
    # Save and output methods
    def _forcing_metadata(self):
        """Record only parameters that affect the selected forcing."""
        metadata = {'forcing_scheme': self.forcing or 'external',
                    'forcing_norm': self.forcing_norm or 'legacy'}
        if self.forcing is not None:
            metadata['forcing_fscale'] = float(self.fscale)
        if (self.forcing_norm == 'amplitude' or self.forcing in ('markov', 'psi_kflow')
                or (self.forcing_norm is None and self.forcing == 'wind')):
            metadata['forcing_famp'] = float(self.famp)
        if self.forcing == 'psi_kflow':
            metadata['forcing_variable'] = 'psi'
            metadata['forcing_amplitude_definition'] = 'peak of Fpsi=famp*sin(fscale*y)'
        if self.forcing_norm in ('enstrophy', 'energy', 'kinetic_energy'):
            metadata['forcing_finput'] = float(self.finput)
        return metadata

    def _validate_model_metadata(self, ds):
        """Legacy files without alpha/damping metadata describe alpha=2 QG."""
        for name, default in (('alpha', 2.), ('gamma', self.gamma), ('beta', self.beta)):
            if float(getattr(ds, name, default)) != float(getattr(self, name)):
                raise ValueError(f'File {name} does not match this model; use a new output directory.')
        old_damping = getattr(ds, 'damping_on', 'vorticity')
        if old_damping != 'q' and not (old_damping == 'vorticity' and self.alpha == 2 and self.gamma == 0):
            raise ValueError('File damping differs from q damping; use a new output directory.')
        for name, value in self._forcing_metadata().items():
            # Old files have no forcing metadata: retain legacy append behavior
            # but do not append a newly normalized run to an unlabelled file.
            default = 'legacy' if name == 'forcing_norm' else value
            if self.forcing == 'psi_kflow':
                default = None  # New forcing requires explicit matching metadata.
            if getattr(ds, name, default) != value:
                raise ValueError(f'File {name} does not match this model; use a new output directory.')

    def _write_model_metadata(self, ds):
        ds.alpha = self.alpha
        ds.damping_on = 'q'
        ds.inversion_definition = 'q_hat=-(k^2+gamma^2)^(alpha/2)*psi_hat; psi_hat[0,0]=0'
        ds.beta_definition = 'Uniform background q gradient; planetary beta only for alpha=2 QG'
        ds.gamma_definition = 'Inverse screening length; alpha!=2 extension is a model choice'
        ds.diagnostic_normalization = 'Grid sums in retained isotropic shells; divide by Nx*Ny for means'
        ds.energy_definition = 'E=-<psi*q>/2, Q=<q^2>/2, Z=<omega^2>/2, K=<|grad psi|^2>/2, before grid-sum scaling; omega=Delta psi'
        for name, value in self._forcing_metadata().items():
            setattr(ds, name, value)
        ds.forcing_normalization_definition = 'Explicit normalization uses full-grid spatial means'

    def create_rst(self,nf,prefix='rst'):
        """Create NetCDF file for model state restart data
        
        Stores full vorticity field and time indices for RK4 or AB3 continuation
        
        Args:
            nf: File counter for restart checkpoint numbering
        """
        outdir = self.savedir
        # Create restart filename with counter (prefix allows ctrl/cda/gnud/ref)
        nc_filename = os.path.join(outdir, "%s_%04d.nc" % (prefix, nf))
            
        if os.path.exists(nc_filename):
            # Append to existing file (auto_complex so 'qrst' reads/writes as complex)
            self.rstds = nc.Dataset(nc_filename, 'a', format='NETCDF4', auto_complex=True)
            try:
                self._validate_model_metadata(self.rstds)
            except ValueError:
                self.rstds.close()
                raise
            self.rst_times = self.rstds.variables['time']
            self.qrst_var = self.rstds.variables['qrst']
            # Guard against silently mixing precisions in one restart file
            # (itemsize also matches the compound r/i representation).
            if self.qrst_var.dtype.itemsize != np.dtype(self.cdtype).itemsize:
                raise ValueError(
                    f"Restart file {nc_filename} stores qrst with itemsize "
                    f"{self.qrst_var.dtype.itemsize}, incompatible with "
                    f"precision={self.precision!r}. Use a fresh savedir.")
        else:
            # Create new file
            self.rstds = nc.Dataset(nc_filename, 'w', format='NETCDF4', auto_complex=True)
            # Create dimensions for time, integration scheme index, and spatial grid
            time_dim = self.rstds.createDimension('time', None) 
            if self.ts_scheme  == 'rk4':
                ind_dim = self.rstds.createDimension('ind', 1) 
            elif self.ts_scheme ==  'ab3':
                ind_dim = self.rstds.createDimension('ind', 3) 
            x_dim = self.rstds.createDimension('x', self.Nx)
            y_dim = self.rstds.createDimension('y', self.Ny)
            # Create coordinate variables
            self.rst_times = self.rstds.createVariable('time', 'f4', ('time',))
            inds = self.rstds.createVariable('ind', 'f4', ('ind',))
            xs = self.rstds.createVariable('x', 'f4', ('x',))
            ys = self.rstds.createVariable('y', 'f4', ('y',))
            # Initialize time step indices based on integration scheme
            if self.ts_scheme  == 'rk4':
                inds[:] = np.array([0,], dtype=np.float32)
            elif self.ts_scheme ==  'ab3':
                inds[:] = np.array([0,1,2], dtype=np.float32)
            # Set spatial coordinates from GPU arrays
            xs[:] = self.x.get()
            ys[:] = self.y.get()

            # Create data variable for the spectral vorticity coefficients.
            # Stored at working precision (c8/c16, Fourier space) so restart is
            # bit-exact, avoiding the FFT round-trip error incurred by saving
            # in physical space.
            rst_ctype = 'c8' if self.precision == 'single' else 'c16'
            self.qrst_var = self.rstds.createVariable('qrst', rst_ctype, ('time','ind', 'y', 'x'), zlib=False)

            # Store simulation parameters as global attributes
            self.rstds.description = "QG Turbulence Simulation RST file"
            self.rstds.rst_space = "spectral"
            self.rstds.precision = self.precision
            self.rstds.dt = self.dt
            self.rstds.Nx = self.Nx
            self.rstds.Ny = self.Ny
            self.rstds.Lx = self.Lx
            self.rstds.Ly = self.Ly
            self.rstds.ts_scheme = self.ts_scheme
            self.rstds.kf = self.fscale
            self.rstds.friction = self.friction
            self.rstds.hyperorder=self.hyperorder
            self.rstds.hyvisc = self.hyvisc
            self.rstds.gamma = self.gamma
            self.rstds.beta = self.beta
            self.rstds.cl = self.cl
            # self.rst_time_offset = 0
        self._write_model_metadata(self.rstds)

    def create_nc(self,nf,prefix='output'):
        """Create NetCDF file for output diagnostics
        
        Initializes file with dimensions and coordinate variables for diagnostics output
        
        Args:
            nf: File counter for output numbering
            prefix: Filename prefix
        """
        outdir = self.savedir
        # Create output filename with counter
        nc_filename = os.path.join(outdir, "%s_%04d.nc"%(prefix,nf))
        
        if os.path.exists(nc_filename):
            # Append to existing file
            self.ds = nc.Dataset(nc_filename, 'a', format='NETCDF4')
            try:
                self._validate_model_metadata(self.ds)
            except ValueError:
                self.ds.close()
                raise
            # Old Z/tz/fz outputs describe q, not relative vorticity.
            # Rename them before creating the distinct relative-vorticity Z outputs.
            scalar_names = [('Ztot', 'Qtot'), ('Zk', 'Qk')]
            for term in ('nlk', 'fk', 'frick', 'visck', 'filtk', 'dak'):
                scalar_names.extend((('tz' + term, 'tq' + term),
                                     ('fz' + term, 'fq' + term)))
            for old, new in scalar_names:
                if old in self.ds.variables and new not in self.ds.variables:
                    self.ds.renameVariable(old, new)
            self.times = self.ds.variables['time']
            self.q_var = self.ds.variables['q']
            self.psi_var = self.ds.variables['psi']
            self.rv_var = self.ds.variables['rv']
            self.Etot_var = self.ds.variables['Etot']
            self.Qtot_var = self.ds.variables['Qtot']
            if 'Ztot' not in self.ds.variables:
                self.ds.createVariable('Ztot', 'f4', ('time',))
            self.Ztot_var = self.ds.variables['Ztot']
            if 'sigma' not in self.ds.variables:
                self.ds.createVariable('sigma', 'f4', ('time', 'k'))
            self.sigma_var = self.ds.variables['sigma']
            # Older files acquire K diagnostics; previous records remain missing.
            if 'Ktot' not in self.ds.variables:
                self.ds.createVariable('Ktot', 'f4', ('time',))
            self.Ktot_var = self.ds.variables['Ktot']
            self.Ek_var = self.ds.variables['Ek']
            self.Qk_var = self.ds.variables['Qk']
            if 'Zk' not in self.ds.variables:
                self.ds.createVariable('Zk', 'f4', ('time', 'k'))
            self.Zk_var = self.ds.variables['Zk']
            if 'Kk' not in self.ds.variables:
                self.ds.createVariable('Kk', 'f4', ('time', 'k'))
            self.Kk_var = self.ds.variables['Kk']
            self.tenlk_var = self.ds.variables['tenlk']
            self.tqnlk_var = self.ds.variables['tqnlk']
            self.tefk_var = self.ds.variables['tefk']
            self.tqfk_var = self.ds.variables['tqfk']
            self.tefrick_var = self.ds.variables['tefrick']
            self.tqfrick_var = self.ds.variables['tqfrick']
            self.tevisck_var = self.ds.variables['tevisck']
            self.tqvisck_var = self.ds.variables['tqvisck']
            self.tefiltk_var = self.ds.variables['tefiltk']
            self.tqfiltk_var = self.ds.variables['tqfiltk']
            self.fenlk_var = self.ds.variables['fenlk']
            self.fqnlk_var = self.ds.variables['fqnlk']
            self.fefk_var = self.ds.variables['fefk']
            self.fqfk_var = self.ds.variables['fqfk']
            self.fefrick_var = self.ds.variables['fefrick']
            self.fqfrick_var = self.ds.variables['fqfrick']
            self.fevisck_var = self.ds.variables['fevisck']
            self.fqvisck_var = self.ds.variables['fqvisck']
            self.fefiltk_var = self.ds.variables['fefiltk']
            self.fqfiltk_var = self.ds.variables['fqfiltk']
            
            self.daF_var = self.ds.variables['daF']
            self.tedak_var = self.ds.variables['tedak']
            self.tqdak_var = self.ds.variables['tqdak']
            self.fedak_var = self.ds.variables['fedak']
            self.fqdak_var = self.ds.variables['fqdak']
        else:
            # Create new file
            self.ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            # Create dimensions for time, spatial grid, and wavenumber spectrum
            time_dim = self.ds.createDimension('time', None) 
            x_dim = self.ds.createDimension('x', self.Nx)
            y_dim = self.ds.createDimension('y', self.Ny)
            k_dim = self.ds.createDimension('k',len(self.kk_iso))
            # Create coordinate variables
            self.times = self.ds.createVariable('time', 'f4', ('time',))
            xs = self.ds.createVariable('x', 'f4', ('x',))
            ys = self.ds.createVariable('y', 'f4', ('y',))
            # Set spatial coordinates from GPU arrays
            xs[:] = self.x.get()
            ys[:] = self.y.get()
            # Create wavenumber coordinate in isotropic spectrum
            kk = self.ds.createVariable('k','f4',('k',))
            kk[:] = self.kk_iso.get()
            ## prognostic variable
            self.q_var = self.ds.createVariable('q', 'f4', ('time', 'y', 'x'), zlib=False)
            self.psi_var = self.ds.createVariable('psi', 'f4', ('time', 'y', 'x'), zlib=False)
            self.rv_var = self.ds.createVariable('rv', 'f4', ('time', 'y', 'x'), zlib=False)
            ## generalized energy, scalar mean square, relative-vorticity enstrophy and deformation
            self.Etot_var = self.ds.createVariable('Etot', 'f4', ('time',))
            self.Qtot_var = self.ds.createVariable('Qtot', 'f4', ('time',))
            self.Ztot_var = self.ds.createVariable('Ztot', 'f4', ('time',))
            self.sigma_var = self.ds.createVariable('sigma', 'f4', ('time', 'k'))
            self.Ktot_var = self.ds.createVariable('Ktot', 'f4', ('time',))
            self.Ek_var = self.ds.createVariable('Ek', 'f4', ('time', 'k'), zlib=False)
            self.Qk_var = self.ds.createVariable('Qk', 'f4', ('time', 'k'), zlib=False)
            self.Zk_var = self.ds.createVariable('Zk', 'f4', ('time', 'k'), zlib=False)
            self.Kk_var = self.ds.createVariable('Kk', 'f4', ('time', 'k'), zlib=False)
            ## tendency budget
            # non-linear advection
            self.tenlk_var = self.ds.createVariable('tenlk', 'f4', ('time', 'k'), zlib=False)
            self.tqnlk_var = self.ds.createVariable('tqnlk', 'f4', ('time', 'k'), zlib=False)
            # forcing 
            self.tefk_var = self.ds.createVariable('tefk', 'f4', ('time', 'k'), zlib=False)
            self.tqfk_var = self.ds.createVariable('tqfk', 'f4', ('time', 'k'), zlib=False)
            # friction
            self.tefrick_var = self.ds.createVariable('tefrick', 'f4', ('time', 'k'), zlib=False)
            self.tqfrick_var = self.ds.createVariable('tqfrick', 'f4', ('time', 'k'), zlib=False)
            # viscosity
            self.tevisck_var = self.ds.createVariable('tevisck', 'f4', ('time', 'k'), zlib=False)
            self.tqvisck_var = self.ds.createVariable('tqvisck', 'f4', ('time', 'k'), zlib=False)
            # filter
            self.tefiltk_var = self.ds.createVariable('tefiltk', 'f4', ('time', 'k'), zlib=False)
            self.tqfiltk_var = self.ds.createVariable('tqfiltk', 'f4', ('time', 'k'), zlib=False)
            ## flux budget
            # non-linear advection
            self.fenlk_var = self.ds.createVariable('fenlk', 'f4', ('time', 'k'), zlib=False)
            self.fqnlk_var = self.ds.createVariable('fqnlk', 'f4', ('time', 'k'), zlib=False)
            # forcing 
            self.fefk_var = self.ds.createVariable('fefk', 'f4', ('time', 'k'), zlib=False)
            self.fqfk_var = self.ds.createVariable('fqfk', 'f4', ('time', 'k'), zlib=False)
            # friction
            self.fefrick_var = self.ds.createVariable('fefrick', 'f4', ('time', 'k'), zlib=False)
            self.fqfrick_var = self.ds.createVariable('fqfrick', 'f4', ('time', 'k'), zlib=False)
            # viscosity
            self.fevisck_var = self.ds.createVariable('fevisck', 'f4', ('time', 'k'), zlib=False)
            self.fqvisck_var = self.ds.createVariable('fqvisck', 'f4', ('time', 'k'), zlib=False)
            # filter
            self.fefiltk_var = self.ds.createVariable('fefiltk', 'f4', ('time', 'k'), zlib=False)
            self.fqfiltk_var = self.ds.createVariable('fqfiltk', 'f4', ('time', 'k'), zlib=False)

            self.daF_var = self.ds.createVariable('daF', 'f4', ('time', 'y', 'x'), zlib=False)
            self.tedak_var = self.ds.createVariable('tedak', 'f4', ('time', 'k'), zlib=False)
            self.tqdak_var = self.ds.createVariable('tqdak', 'f4', ('time', 'k'), zlib=False)
            self.fedak_var = self.ds.createVariable('fedak', 'f4', ('time', 'k'), zlib=False)
            self.fqdak_var = self.ds.createVariable('fqdak', 'f4', ('time', 'k'), zlib=False)

            self.ds.description = "QG Turbulence Simulation"
            self.ds.dt = self.dt
            self.ds.Nx = self.Nx
            self.ds.Ny = self.Ny
            self.ds.Lx = self.Lx
            self.ds.Ly = self.Ly
            self.ds.ts_scheme = self.ts_scheme
            self.ds.kf = self.fscale
            self.ds.hyperorder=self.hyperorder
            self.ds.friction = self.friction
            self.ds.hyvisc = self.hyvisc
            self.ds.gamma = self.gamma
            self.ds.beta = self.beta
            self.ds.cl = self.cl
            # self.nc_time_offset = 0
        self._write_model_metadata(self.ds)
        for var in (self.Ktot_var, self.Kk_var):
            var.long_name = 'Kinetic energy in legacy grid-sum shell normalization'
            var.units = '1'
        for name in ('Qtot', 'Qk'):
            self.ds[name].long_name = 'Half active-scalar mean square in grid-sum shell normalization'
        for name in ('Ztot', 'Zk'):
            self.ds[name].long_name = 'Relative-vorticity enstrophy in grid-sum shell normalization'
        self.sigma_var.long_name = 'Cumulative deformation rate sqrt(cumsum(Zk)/(Nx*Ny))'
        self.sigma_var.units = '1/time'

    def save_rst(self,it):
        """Save model state to restart file for integration continuation

        Stores the spectral vorticity coefficients (and the RHS history needed
        for AB3) directly in Fourier space, so the restart is bit-exact at
        working precision rather than incurring an FFT round-trip error.

        Args:
            it: Output record number index
        """
        # Copy spectral coefficients to host at working precision (no FFT round trip).
        q_c = self.q_hat.get().astype(self.cdtype)
        k1_p_c = self.k1_p.get().astype(self.cdtype)
        k1_pp_c = self.k1_pp.get().astype(self.cdtype)
        # Record simulation time
        self.rst_times[it] = self.t
        # Store time history based on integration scheme
        if self.ts_scheme == 'ab3':
            # AB3 requires 3 previous steps
            self.qrst_var[it,0,:,:] = k1_pp_c
            self.qrst_var[it,1,:,:] = k1_p_c
            self.qrst_var[it,2,:,:] = q_c
        elif self.ts_scheme == 'rk4':
            # RK4 only requires current state
            self.qrst_var[it,0,:,:] = q_c
        # Flush to disk
        self.rstds.sync()
        del q_c, k1_p_c, k1_pp_c
        gc.collect()


    def save_var(self,it):
        """Save diagnostic variables to output file
        
        Computes and stores spectral energy/scalar-variance and all budget terms
        
        Args:
            it: Output record number index
        """
        self.force_q = self._forcing_at_state(self.q_hat, self.p_hat, self.t)
        # Transform prognostic fields to physical space
        p_r = ifft2(self.p_hat).real.get()
        q_r = ifft2(self.q_hat).real.get()
        rv_r = ifft2(self.rv_hat).real.get()
        # Record simulation time
        self.times[it] = self.t
        # Store prognostic variables
        self.q_var[it,:,:] = q_r
        self.psi_var[it,:,:] = p_r
        self.rv_var[it,:,:] = rv_r
        # Store diagnostic variables
        # Energy, scalar mean square and relative-vorticity enstrophy
        self.Etot_var[it] = self.get_Etot(self.p_hat)
        self.Qtot_var[it] = self.get_Qtot(self.q_hat)
        z_kk = self.get_Zk(self.rv_hat)
        self.Ztot_var[it] = np.sum(z_kk)
        self.sigma_var[it,:] = self.get_sigma(z_kk)
        self.Ek_var[it,:] = self.get_Ek(self.p_hat) 
        self.Qk_var[it,:] = self.get_Qk(self.q_hat)
        self.Zk_var[it,:] = z_kk
        self.Ktot_var[it] = self.get_Ktot(self.p_hat)
        self.Kk_var[it,:] = self.get_Kk(self.p_hat)
        # Generalized-energy and scalar-variance budget terms
        # Non-linear advection transfer and flux
        self.tenlk_var[it,:],self.tqnlk_var[it,:], self.fenlk_var[it,:], self.fqnlk_var[it,:] = self.get_diagNL(self.p_hat,self.q_hat)
        # Forcing transfer and flux
        self.tefk_var[it,:], self.tqfk_var[it,:], self.fefk_var[it,:], self.fqfk_var[it,:] = self.get_diagF(self.p_hat,self.q_hat,self.force_q)
        # Friction dissipation and flux
        self.tefrick_var[it,:], self.tqfrick_var[it,:],self.fefrick_var[it,:], self.fqfrick_var[it,:] = self.get_diagFric(self.p_hat,self.q_hat)
        # Hyperviscosity dissipation and flux
        self.tevisck_var[it,:], self.tqvisck_var[it,:], self.fevisck_var[it,:], self.fqvisck_var[it,:] = self.get_diagVisc(self.p_hat,self.q_hat)
        # Spectral filter loss and flux
        self.tefiltk_var[it,:], self.tqfiltk_var[it,:],self.fefiltk_var[it,:], self.fqfiltk_var[it,:] = self.get_diagFilt(self.p_hat,self.q_hat)

        # Save DA term
        self.daF_var[it, :, :] = ifft2(self.da_term).real.get()
        # Save DA diagnostics
        self.tedak_var[it,:], self.tqdak_var[it,:], self.fedak_var[it,:], self.fqdak_var[it,:] = self.get_diagDa(self.p_hat, self.q_hat, self.da_term)

        # Flush to disk
        self.ds.sync()
        del q_r, p_r, rv_r
        gc.collect()
        
    # Plotting and visualization methods
    def plot_diag(self, save_path=None):
        """Plot energy spectrum and budget diagnostics
        
        Visualizes PV field, energy spectrum, energy tendency budget, and flux budget
        
        Args:
            save_path: Optional path to save figure
        """
        self.force_q = self._forcing_at_state(self.q_hat, self.p_hat, self.t)
        # Compute all diagnostic budget terms
        Ek = self.get_Ek(self.p_hat)
        
        # Non-Linear advection transfer & flux
        tenl, _, fenl, _ = self.get_diagNL(self.p_hat, self.q_hat)
        
        # Forcing (injection) transfer & flux
        teF, _, feF, _ = self.get_diagF(self.p_hat, self.q_hat, self.force_q)
        
        # Dissipation terms (all dissipative processes)
        tevisc, _, fevisc, _ = self.get_diagVisc(self.p_hat, self.q_hat)
        tefric, _, fefric, _ = self.get_diagFric(self.p_hat, self.q_hat)
        tefilt, _, fefilt, _ = self.get_diagFilt(self.p_hat, self.q_hat)
        
        teda = cp.zeros_like(tenl)
        feda = cp.zeros_like(fenl)
        teda,_,feda,_ = self.get_diagDa(self.p_hat,self.q_hat,self.da_term)
        
        # Normalize to mean per grid point
        norm_fac = 1.0 / (self.Nx * self.Ny)
        
        # Scale energy tendencies
        tenl *= norm_fac
        teF *= norm_fac
        tevisc *= norm_fac
        tefric *= norm_fac
        tefilt *= norm_fac
        teda *= norm_fac
        
        # Scale energy fluxes
        fenl *= norm_fac
        feF *= norm_fac
        fevisc *= norm_fac
        fefric *= norm_fac
        fefilt *= norm_fac
        feda *= norm_fac
        
        # Scale energy spectrum
        Ek = Ek * norm_fac

        # Compute residual budget terms
        # Tendency residual: should sum to zero in steady state
        te_sum = tenl + teF + tevisc + tefric + tefilt + teda
        
        # Flux residual: cumulative sum of all fluxes
        fe_sum = fenl + feF + fevisc + fefric + fefilt + feda

        # Create figure with subplots
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(30, 18), tight_layout=True)
        gs = fig.add_gridspec(6, 6)

        # Assign subplot positions
        ax_pv = fig.add_subplot(gs[0:4, 1:5])
        ax_spec = fig.add_subplot(gs[4:, 0:2])
        ax_tendency = fig.add_subplot(gs[4:, 2:4]) 
        ax_flux = fig.add_subplot(gs[4:, 4:])      

        # Plotting diagnostics

        # Panel 1: PV Field
        q_phys = ifft2(self.q_hat).real.get()
        im = ax_pv.imshow(q_phys, cmap=self.my_div,vmin=-10,vmax=10,
                          extent=[0, self.Lx, 0, self.Ly])
        ax_pv.set_title(f'Active scalar q (alpha={self.alpha:g}, t={self.t:.2f})', fontsize=30, fontweight='bold')
        ax_pv.set_xlabel('x')
        ax_pv.set_ylabel('y')
        cbar = fig.colorbar(im, ax=ax_pv, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('q')

        # Common Wavenumber Axis 
        ks = self.kk_iso.get() /(2*cp.pi/self.Lx)

        # Panel 2: Energy Spectrum
        ax_spec.loglog(ks, Ek, color='tab:blue', linewidth=3, label='Energy Spec')
        # References
        ks_direct = np.array([18., 80.]) /(2*cp.pi/self.Lx)
        ks_inv = np.array([5., 16.]) /(2*cp.pi/self.Lx)
        ax_spec.axvline(self.fscale/(2*cp.pi/self.Lx), color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        if self.alpha == 2 and self.gamma == 0:
            ax_spec.loglog(ks_direct, 0.5 * ks_direct**-3, 'k--', label='$k^{-3}$', alpha=0.6)
            ax_spec.loglog(ks_inv, 0.1 * ks_inv**-(5/3), 'k-.', label='$k^{-5/3}$', alpha=0.6)
        
        ax_spec.set_title('Generalized energy spectrum', fontweight='bold')
        ax_spec.set_xlabel('Wavenumber $k$')
        ax_spec.set_ylabel('$E(k)$')
        ax_spec.set_xlim([1, int(self.Nx/2)])
        ax_spec.set_ylim([1e-20,1])
        ax_spec.grid(True, which='both', linestyle='--', alpha=0.3)
        ax_spec.legend(loc='lower left', fontsize='small')

        # Panel 3: Energy Tendency Budget (Rate of Change)
        # Plots separate lines for Viscosity (High k) and Friction (Low k)
        ax_tendency.axvline(16, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_tendency.semilogx(ks, tenl, label='NL Transfer', color='tab:blue', linewidth=2.5)
        ax_tendency.semilogx(ks, teF, label='Forcing', color='tab:green', linewidth=2.5)
        ax_tendency.semilogx(ks, tevisc, label='Viscosity', color='tab:orange', linewidth=2.5)
        ax_tendency.semilogx(ks, tefric, label='Friction', color='tab:brown', linewidth=2.5)
        ax_tendency.semilogx(ks, tefilt, label='Filter', color='tab:purple', linewidth=2.5)
        ax_tendency.semilogx(ks, teda, label='DA', color='tab:red', linewidth=2.5)
        ax_tendency.semilogx(ks, te_sum, 'k--', label='Sum (Residual)', linewidth=1.5)
        
        ax_tendency.axhline(0, color='k', linestyle='-', linewidth=1.5)
        ax_tendency.set_xlim([1, int(self.Nx/2)])
        ax_tendency.set_title('Energy Tendency Budget ($dE/dt$)', fontweight='bold')
        ax_tendency.set_xlabel('Wavenumber $k$')
        ax_tendency.set_ylabel('Rate')
        ax_tendency.grid(True, which='both', linestyle='--', alpha=0.3)
        ax_tendency.legend(fontsize='small', loc='best')

        # Panel 4: Energy Flux Budget (Cumulative Transfer)
        ax_flux.axvline(16, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_flux.semilogx(ks, fenl, label='NL Flux $\Pi_{NL}$', color='tab:blue', linewidth=2.5)
        ax_flux.semilogx(ks, feF, label='Forcing Flux', color='tab:green', linewidth=2.5)
        ax_flux.semilogx(ks, fevisc, label='Visc. Flux', color='tab:orange', linewidth=2.5)
        ax_flux.semilogx(ks, fefric, label='Fric. Flux', color='tab:brown', linewidth=2.5)
        ax_flux.semilogx(ks, fefilt, label='Filt. Flux', color='tab:purple', linewidth=2.5)
        ax_flux.semilogx(ks, feda, label='DA Flux', color='tab:red', linewidth=2.5)
        ax_flux.semilogx(ks, fe_sum, 'k--', label='Sum (Residual)', linewidth=1.5)

        ax_flux.axhline(0, color='k', linestyle='-', linewidth=1.5)
        ax_flux.set_xlim([1, int(self.Nx/2)])
        ax_flux.set_title('Energy Flux Budget ($\Pi_E$)', fontweight='bold')
        ax_flux.set_xlabel('Wavenumber $k$')
        ax_flux.set_ylabel('Flux')
        ax_flux.grid(True, which='both', linestyle='--', alpha=0.3)
        ax_flux.legend(fontsize='small', loc='best')

        # --- 5. Save or Show ---
        if save_path:
            plt.savefig(save_path, dpi=100) # dpi=100 is fast for frames
            plt.close(fig)
        else:
            plt.show()

    def save_snapshot(self,nstep):
        """Save diagnostic figure as PNG snapshot during simulation
        
        Creates figures directory and stores plot at specified timestep
        
        Args:
            nstep: Integration step number for figure naming
        """
        # Create figures output directory
        outdir = os.path.join(self.savedir, "figs")
        os.makedirs(outdir, exist_ok=True)
        
        # Save figure with timestep counter
        filename = os.path.join(outdir, f"snap_{nstep:04d}.png")
        self.plot_diag(save_path=filename)

    # Main simulation loop
    def run(self,scheme='ab3',tmax=40,tsave=200,tsave_rst=2000,nsave=100,savedir='run_0',saveplot=False):
        """Run main simulation loop with output checkpointing
        
        Integrates model forward in time with periodic diagnostics and restart saves
        
        Args:
            scheme: Time stepping scheme ('ab3' or 'rk4')
            tmax: Maximum simulation time
            tsave: Steps between diagnostic output
            nsave: Diagnostics saved per output file
            savedir: Directory for output files
            saveplot: Whether to save plotting snapshots
        """
        self.ts_scheme = scheme
        self.tmax = tmax
        self.tsave = tsave
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        self.tsave_rst = tsave_rst
        nrst = nsave

        # Initialize or continue from restart time
        if self.is_not_rst:
            self.t = self.rdtype(0.0)
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
            self.t = self.trst
            # Calculate which file and position to resume from
            n_start_idx = int(round(self.trst / self.dt))
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
        

        # Main time integration loop
        for n in range(n_start, int(tmax/self.dt)+1):
            self.n_steps = n
            # print diagnostics to console every 10,000 steps
            if n%10000 == 0:
                # Compute and print energy and diagnostic statistics
                E_crt = self.get_Etot(self.p_hat)/self.Nx/self.Ny
                Vrms_crt = self.get_Vrms(self.p_hat)
                Qrms_crt = self.get_Qrms(self.q_hat)
                import time
                print(f"Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.n_steps:7d}  t={self.t:9.6f}s E={E_crt:.4e} Vrms={Vrms_crt:.4e} Qrms={Qrms_crt:.4e}", end="\n")
            # Check if need to create new output file
            if insave == nsave:
                itsave = 0
                if nf > nf0:
                    self.ds.close()
                self.create_nc(nf)
                insave = 0
                nf += 1
            # Save diagnostics at specified intervals
            if n%tsave == 0:
                self.save_var(itsave)
                import time
                print(f"[save_var] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.n_steps:7d}  t={self.t:9.6f}s ", end="\n")
                if saveplot:
                    self.plot_diag()
                itsave +=1
                insave +=1

            # Check if need to create new restart file
            if self.n_steps % tsave_rst==0:
                if inrst == nrst:
                    itrst = 0
                    if nfrst > nfrst0:
                        self.rstds.close()
                    self.create_rst(nfrst)
                    inrst = 0
                    nfrst += 1

                self.save_rst(itrst)
                import time
                print(f"[save_rst] Local time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}   step {self.n_steps:7d}  t={self.t:9.6f}s ", end="\n")
                itrst += 1
                inrst += 1
            # Advance model state by one timestep
            self._step_forward()
            self.t = (self.n_steps+1) * self.dt
        # Close output files at end of simulation
        self.ds.close()
        self.rstds.close()
        print('Done.')
    
    def _my_div(self):
        my_div_color = np.array(  [
                 [0,0,123],
                [9,32,154],
                [22,58,179],
                [34,84,204],
                [47,109,230],
                [63,135,247],
                [95,160,248],
                [137,186,249],
                [182,213,251],
                [228,240,254],
                [255,255,255],
                [250,224,224],
                [242,164,162],
                [237,117,113],
                [235,76,67],
                [233,52,37],
                [212,45,31],
                [188,39,26],
                [164,33,21],
                [140,26,17],
                [117,20,12]
                ])/255
        self.my_div = LinearSegmentedColormap.from_list('div',my_div_color, N = 256)
