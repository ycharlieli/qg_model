
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
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

class QGModel:
    """2D Quasi-Geostrophic turbulence model with spectral methods"""
    def __init__(self, Nx, Ny, Lx=2*cp.pi, Ly=2*cp.pi, dt=0.001,
                 beta=0, gamma=0, 
                 friction=0.01,k_friction = 20000,visc2 = 0,hyperorder=1,sp_filtr=False,cl=0,
                 forcing=None,fscale=4,finput=3,famp=1.):
        """Initialize QG model with grid and parameters
        
        Args:
            Nx, Ny: Grid dimensions
            Lx, Ly: Domain sizes
            dt: Time step
            beta: Beta parameter (Coriolis gradient)
            gamma: Stratification parameter
            friction: Friction coefficient
            k_friction: Wavenumber cutoff for friction
            visc2: Eddy viscosity coefficient
            hyperorder: Order of hyperviscosity (1=Laplacian, 2=Biharmonic, etc.)
            sp_filtr: Whether to apply spectral filter
            cl: Leith parameter for biharmonic viscosity
            forcing: Forcing type ('wind', 'thuburn', 'markov', None)
            fscale: Forcing wavenumber scale
            finput: Forcing enstrophy injection rate
            famp: Forcing amplitude
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self._init_grid()
        # QG parameters
        self.beta = beta # Coriolis gradient
        self.gamma = gamma # stratification parameter
        self.friction = friction # large scale friction
        self.k_friction = k_friction # wavenumber cutoff for friction
        self.visc2 = visc2 # eddy viscosity
        if hyperorder == 1:
            self.hyvisc = self.visc2
        else:
            self.hyvisc = 10*1/(self.Nx/2*(2*cp.pi/self.Lx))**(hyperorder*2) # viscosity t_eddy/kmax**(2n)
        self.hyperorder = hyperorder # order of hyper viscosity, 1-> Newnation 2-> biharmonic ...
        self.sp_filtr = sp_filtr # spectral filter impose on the tail of spectral (Arbic 2003)
        self.cl = cl #leith parameter
        self.forcing = forcing
        self.fscale = fscale # scale of wind
        self.finput = finput # enstrophy injection rate of wind
        self.famp = famp
        self.force_q = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self.cda_term = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self.k1_p = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self.k1_pp = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self.is_not_rst = True
        self._prebuild_operator()
        self._my_div()
        # gpu or cpu?  backend
### private term
    def _init_grid(self):
        """Initialize computational grid and wavenumber arrays"""
        self.x = cp.linspace(0,self.Lx,self.Nx)
        self.y = cp.linspace(0,self.Ly,self.Ny)
        self.x2d, self.y2d = cp.meshgrid(self.x,self.y)
        # Grid indices for FFT
        nx = cp.arange(self.Nx); nx[int(self.Nx/2):] -= self.Nx # shift for proper wavenumbers
        ny = cp.arange(self.Ny); ny[int(self.Ny/2):] -= self.Ny
        self.nx2d, self.ny2d = cp.meshgrid(nx,ny)
        # Wavenumbers
        kx = 2*cp.pi*nx/self.Lx
        ky = 2*cp.pi*ny/self.Ly
        # 2D wavenumber arrays
        self.kx2d, self.ky2d = cp.meshgrid(kx,ky) 
        # 2D isotropic wavenumber magnitude
        self.kk = cp.sqrt(self.kx2d**2+self.ky2d**2) 

        # self.kk_intvl = cp.round(self.kk).astype('int')
        # self.kk_set = cp.unique(self.kk_intvl) # isotropic wavenumber magnitude
        # self.kk_range  = self.kk_set < cp.sqrt(cp.min(cp.array([(self.kx2d**2).max(),(self.ky2d**2).max()])))
        # self.kk_iso = self.kk_set[self.kk_range]
        k_bins_grid = cp.round(cp.sqrt(self.nx2d**2 + self.ny2d**2)).astype('int')
        self.kk_idx_set, self.kk_idx = cp.unique(k_bins_grid, return_inverse=True)
        self.kk_set = self.kk_idx_set * (2*cp.pi/self.Lx)
        self.kk_range = self.kk_idx_set < int(self.Nx/2)
        self.kk_iso = self.kk_set[self.kk_range]
    def _init_friction(self):
        """Initialize friction mask for large-scale modes"""
        self.friction_mask = cp.zeros_like(self.kk)
        self.friction_mask[self.kk <= self.k_friction] = 1 # apply friction to large scales
        self.friction_mask[0,0] = 1 # handle mean flow

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
        #inversion of poisson/helmholtz equation
        self.inversion = 1/(-self.kx2d**2-self.ky2d**2-self.gamma**2) 
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
        
        self._init_friction()
        # preallocate for jacobian  for dealiasing
        self.Nxpad = int(3*self.Nx/2)
        self.Nypad = int(3*self.Ny/2)
        self.pad_buffer = 1j*cp.zeros((self.Nxpad,self.Nypad))

        self.parseval_fac = (self.Nxpad*self.Nypad)/(self.Nx*self.Ny)
        # for truncating in padded field
        self.px0 = int((self.Nxpad-self.Nx)/2)
        self.px1= self.px0+self.Nx
        self.py0 = int((self.Nypad-self.Ny)/2)
        self.py1 = self.py0+self.Ny
        
        
    def _padding(self,ft):
        self.pad_buffer.fill(0j)
        #shift zero frequency to center
        self.pad_buffer[self.px0:self.px1,self.py0:self.py1]=fftshift(ft)  
        # shift back than the low frequency will back to the edge 
        # garuntee the power is unchange to return the same value, i.e. parseval theorem
        ft_pad = fftshift(self.parseval_fac*self.pad_buffer) 
        return ft_pad

    def _unpadding(self, ft_pad):
        
        ft_shift = fftshift(ft_pad)[self.px0:self.px1,self.py0:self.py1]
        ft = fftshift(ft_shift/self.parseval_fac)
        
        return ft

        
    def _get_rhs(self,q_hat):
        """Compute right-hand side of QG dynamics equation
        
        Returns the tendency from all processes: advection, beta effect, damping
        """
        p_hat = self.inversion*q_hat # invert to get streamfunction
        rv_hat = q_hat + self.gamma**2*p_hat # relative vorticity
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        beta_term = self.beta*self.kx2d*1j*p_hat
        damping_term = (-self.friction_mask*self.friction + self.hylap)*rv_hat
        damping_term+=self._compute_leith_term(rv_hat)
        
        return -jacobian_term-beta_term+damping_term+self.force_q+self.cda_term

    def _rk4(self, q_hat):
        """Compute 4th order Runge-Kutta stages"""
        k1 = self._get_rhs(q_hat)
        k2 = self._get_rhs(q_hat + 0.5 * self.dt * k1)
        k3 = self._get_rhs(q_hat + 0.5 * self.dt * k2)
        k4 = self._get_rhs(q_hat + self.dt * k3)

        return k1,k2,k3,k4

    def _step_forward(self):
        """Advance simulation one time step using specified scheme"""
        if self.forcing == 'wind':
            self._set_windforce()
        elif self.forcing =='thuburn':
            self.force_q = fft2(0.1*cp.sin(32*np.pi*self.x2d))
        elif self.forcing == 'markov':
            self._update_markovforce()
        q = self.q_hat

        if self.ts_scheme=='rk4':
            # 4th order Runge-Kutta integration
            k1,k2,k3,k4 = self._rk4(q)
            self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        elif self.ts_scheme == 'ab3':
            # 3rd order Adams-Bashforth integration
            if self.is_not_rst:
                if self.n_steps == 0:
                    k1,k2,k3,k4 = self._rk4(q)
                    self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    self.k1_pp = k1.copy() # store RHS history for AB3
                elif self.n_steps == 1:
                    k1,k2,k3,k4 = self._rk4(q)
                    self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    self.k1_p = k1.copy()
                else:
                    k1 = self._get_rhs(q)
                    self.q_hat = q + self.dt/12*(23*k1-16*self.k1_p+5*self.k1_pp)
                    self.k1_pp = self.k1_p
                    self.k1_p = k1.copy()
            else:
                k1 = self._get_rhs(q)
                self.q_hat = q + self.dt/12*(23*k1-16*self.k1_p+5*self.k1_pp)
                self.k1_pp = self.k1_p
                self.k1_p = k1.copy()

        self.q_hat *=self.filtr # apply spectral filter
        self.p_hat = self.inversion*self.q_hat
        self.rv_hat = self.q_hat + self.gamma**2*self.p_hat
        
    
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

    def _compute_leith_term(self, rv_hat=None):
        """Compute biharmonic Leith viscosity term
        
        Args:
            rv_hat: Relative vorticity in Fourier space (uses current value if None)
        """
        if rv_hat is None:
            rv_hat = self.rv_hat # use current value for stable stepping
        dxrv_hat = self.kx2d*1j*rv_hat
        dyrv_hat = self.ky2d*1j*rv_hat
        
        dxrv_r = ifft2(self._padding(dxrv_hat)).real
        dyrv_r = ifft2(self._padding(dyrv_hat)).real
        
        grad_rv_r = cp.sqrt(dxrv_r**2+dyrv_r**2)
        
        nabla = self.Lx/self.Nx
        nu_e = (self.cl*nabla)**3*grad_rv_r # eddy viscosity
        
        flux_x_r = nu_e*dxrv_r
        flux_y_r = nu_e*dyrv_r

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
            self.rv_hat = self.lap*self.p_hat
            # initial potential vorticity (q)
            self.q_hat = self.rv_hat - self.gamma**2*self.p_hat
            # normalize mean of total energy to 0.5
            ene_tot = self.get_Etot(self.p_hat)
            norm_fac = cp.sqrt(self.eini/(ene_tot/(self.Nx*self.Ny)))
            self.p_hat = norm_fac*self.p_hat
            # initial relavitve vorticity (rv)
            self.rv_hat = self.lap*self.p_hat
            # initial potential vorticity (q)
            self.q_hat = self.rv_hat - self.gamma**2*self.p_hat
        
    def set_initial_condition(self,scheme='gauss',eini=0,q_ini=None,trst=0):
        self.eini = eini
        self.trst = trst
        if scheme == 'jcm1984':
            k_peak = 6
            # kk**(-A)*(1 + (kk/k0)**4)**(-B)
            # generate Fourier conponent of initial streamfunction field
            ls=3
            ss=-3
            A = 3-ls
            B = (3-ss-A)/4
            amp = cp.sqrt(self.kk**(-A)*(1 + (self.kk/self.k_peak)**4)**(-B))
            rand_p = fft2(cp.random.randn(*self.kk.shape))
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
            psi_phys = cp.random.randn(self.Nx, self.Ny)
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
            self.q_hat = self.rv_hat - self.gamma**2 * self.p_hat
        elif scheme == 'rst':
            # give initial q field manually
            # if q_ini[:,:,0].shape[0] > self.Nx:
            #     # high res ini spectral cut
            #     q_ini = self.spec_cut(q_ini)
                


            if q_ini.ndim == 2:

                self.q_hat = fft2(cp.array(q_ini.copy()))
                self.p_hat = self.inversion*self.q_hat
                self.rv_hat= self.lap*self.p_hat
            elif q_ini.ndim == 3 :
                self.is_not_rst = False
                self.q_hat = fft2(cp.array(q_ini[2,:,:].squeeze()))
                self.p_hat = self.inversion*self.q_hat
                self.rv_hat= self.lap*self.p_hat
                self.k1_p = fft2(cp.array(q_ini[1,:,:].squeeze()))
                self.k1_pp  = fft2(cp.array(q_ini[0,:,:].squeeze()))
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
        self.Etot = self.get_Etot(self.p_hat)
        self.Ek = self.get_Ek(self.p_hat)
        self.tenlk, self.tznlk, self.fenlk, self.fznlk = self.get_diagNL(self.p_hat,self.q_hat)   
     
### forcing term 
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
        
        # inputF = cp.sum(cp.real(cp.conj(self.q_hat)*Fq_hat))/(self.Nx*self.Ny)**2 # current enstrophy injection?
        # inputF = -cp.sum(cp.real(cp.conj(self.p_hat) * Fq_hat)) / (self.Nx * self.Ny)**2 # energy injection
        # norm_fac = 1.*(self.finput)/(inputF)
        # Fq_hat *= norm_fac
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
            self.fR = 0.0
        else:
            self.fR = cp.exp(-self.dt / t_r)
        self.fseed = 10
        #  Pre-calculate the amplitude coefficient: A * sqrt(1 - R^2)
        # This ensures the variance of the forcing stays constant at A^2 over time.
        self.fcoef = norm_fac*famp * cp.sqrt(1 - self.fR**2)
        
        # Initialize the forcing field F_{n-1} to zero
        self.force_q = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)

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
        noise_phys = cp.random.randn(self.Nx, self.Ny)
        noise_hat = fft2(noise_phys)
        
        # Apply spectral mask to specific wavenumber shell
        noise_hat *= self.fmask
        
        # Markov update with temporal correlation (F_n = a*noise + r*F_{n-1})
        self.force_q = (self.fcoef * noise_hat) + (self.fR * self.force_q)

## diagnostic term
    def get_Ek(self,p_hat):
        """Compute isotropic energy spectrum
        
        Integrates energy density in spectral shells using Parseval's theorem
        """
        # Energy density in spectral space
        ene_dens = 0.5*(self.kk**2+self.gamma**2)*cp.abs(p_hat)**2
        # Physical space using Parseval's Theorem
        norm_fac = 1/(self.Nx*self.Ny)
        ene_kk = npg.aggregate(self.kk_idx.ravel().get(),ene_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ene_kk
    def get_Etot(self,p_hat):
        """Compute total energy"""
        ene_kk = self.get_Ek(p_hat)
        ene_tot = np.sum(ene_kk) 
        return ene_tot

    def get_Vrms(self,p_hat):
        """Compute RMS velocity"""
        ene_tot = self.get_Etot(p_hat)
        vrms = np.sqrt(2*ene_tot/(self.Nx*self.Ny))
        return vrms
    def get_Qrms(self,q_hat):
        """Compute RMS potential vorticity"""
        ens_tot = self.get_Ztot(q_hat)
        qrms = np.sqrt(2*ens_tot/(self.Nx*self.Ny))
        return qrms
    def get_Zk(self,q_hat):
        """Compute isotropic enstrophy spectrum"""
        # Enstrophy density in spectral space
        ens_dens = 0.5*np.abs(q_hat)**2
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic enstrophy spectrum in physical space using Parseval's Theorem
        ens_kk = npg.aggregate(self.kk_idx.ravel().get(),ens_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ens_kk
    def get_Ztot(self,q_hat):
        """Compute total enstrophy"""
        ens_kk = self.get_Zk(q_hat)
        ens_tot = np.sum(ens_kk)
        return ens_tot
    def get_TENL(self,p_hat,q_hat):
        """Compute spectral energy transfer from non-linear advection"""
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # Energy transfer: T_E = -Re(p* * J)
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        return tenl
    
    def get_TZNL(self,p_hat,q_hat):
        """Compute spectral enstrophy transfer from non-linear advection"""
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # Enstrophy transfer: T_Z = -Re(q* * J)
        tznl = -cp.real(cp.conj(q_hat)*jacobian_term)
        return tznl

    def get_diagNL(self,p_hat,q_hat):
        """Compute energy and enstrophy budgets from non-linear advection
        
        Returns spectral transfer and flux for both quantities
        """
        jacobian_term = self._compute_jacobian(p_hat,q_hat)
        # spectral energy transfer of non-linear advection
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        # spectral enstrophy transfer of non-linear advection
        tznl = -cp.real(cp.conj(q_hat)*jacobian_term)

        # Isotropic spectrum of energy transfer of non-linear advection 
        # In physical space using spectral aggregation
        norm_fac = 1/(self.Nx*self.Ny)
        tenl_kk = npg.aggregate(self.kk_idx.ravel().get(),tenl.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of non-linear advection
        # In physical space 
        tznl_kk = npg.aggregate(self.kk_idx.ravel().get(),tznl.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of non-linear advection
        fenl_kk = -np.cumsum(tenl_kk)
        # Isotropic spectrum of enstrophy flux of non-linear advection
        fznl_kk = -np.cumsum(tznl_kk)

        return tenl_kk, tznl_kk, fenl_kk, fznl_kk

    def get_diagF(self,p_hat,q_hat,force_q):
        """Compute energy and enstrophy budgets from forcing
        
        Returns spectral transfer and flux for both quantities
        """
        # spectral energy transfer of forcing
        teF = -cp.real(cp.conj(p_hat)*force_q)
        # spectral enstrophy transfer of forcing
        tzF = cp.real(cp.conj(q_hat)*force_q)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer 
        # In physical space using spectral aggregation
        teF_kk = npg.aggregate(self.kk_idx.ravel().get(),teF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer 
        # In physical space 
        tzF_kk =npg.aggregate(self.kk_idx.ravel().get(),tzF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of forcing
        feF_kk = np.cumsum(teF_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of forcing
        fzF_kk = np.cumsum(tzF_kk[::-1])[::-1]

        return teF_kk, tzF_kk, feF_kk, fzF_kk

    def get_diagCda(self,p_hat,q_hat,cda_term):
        """Compute energy and enstrophy budgets from data assimilation nudging
        
        Returns spectral transfer and flux for both quantities
        """
        # spectral energy transfer of CDA
        tecda = -cp.real(cp.conj(p_hat)*cda_term)
        # spectral enstrophy transfer of CDA
        tzcda = cp.real(cp.conj(q_hat)*cda_term)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer 
        # In physical space using spectral aggregation
        tecda_kk = npg.aggregate(self.kk_idx.ravel().get(),tecda.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer 
        # In physical space 
        tzcda_kk =npg.aggregate(self.kk_idx.ravel().get(),tzcda.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux
        fecda_kk = np.cumsum(tecda_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux
        fzcda_kk = np.cumsum(tzcda_kk[::-1])[::-1]

        return tecda_kk, tzcda_kk, fecda_kk, fzcda_kk

    def get_diagFric(self,p_hat,q_hat):
        """Compute energy and enstrophy dissipation from large-scale friction
        
        Returns spectral dissipation and flux for both quantities
        """
        # spectral energy transfer of friction
        tefric = -2*self.friction_mask*self.friction* 0.5*self.kk**2*cp.abs(p_hat)**2
        # spectral enstrophy transfer of friction
        tzfric = -2*self.friction_mask*self.friction* 0.5*cp.abs(q_hat)**2
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of friction
        # In physical space using spectral aggregation
        tefric_kk = npg.aggregate(self.kk_idx.ravel().get(),tefric.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of friction
        # In physical space 
        tzfric_kk = npg.aggregate(self.kk_idx.ravel().get(),tzfric.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of friction
        fefric_kk = np.cumsum(tefric_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of friction
        fzfric_kk = np.cumsum(tzfric_kk[::-1])[::-1]
        return tefric_kk, tzfric_kk, fefric_kk, fzfric_kk

    def get_diagVisc(self, p_hat, q_hat):
        """Compute energy and enstrophy dissipation from hyperviscosity
        
        Returns spectral dissipation and flux for both quantities
        """
        rv_hat = q_hat + self.gamma**2*p_hat
        # spectral energy transfer of viscosity
        tevisc = -cp.real(cp.conj(p_hat)*(self.hylap*rv_hat+self._compute_leith_term(rv_hat)))
        # spectral enstrophy transfer of viscosity
        tzvisc = cp.real(cp.conj(q_hat)*(self.hylap*rv_hat+self._compute_leith_term(rv_hat)))
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of viscosity
        # In physical space using spectral aggregation
        tevisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tevisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of viscosity
        # In physical space 
        tzvisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tzvisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of viscosity
        fevisc_kk = np.cumsum(tevisc_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of viscosity
        fzvisc_kk = np.cumsum(tzvisc_kk[::-1])[::-1]
        return tevisc_kk, tzvisc_kk, fevisc_kk, fzvisc_kk

    def get_diagFilt(self,p_hat,q_hat):
        """Compute energy and enstrophy loss from spectral filter
        
        Returns spectral dissipation and flux for both quantities
        """
        filt_rate = (self.filtr - 1.) / self.dt
        # spectral energy transfer of filter
        tefilt = -cp.real(cp.conj(p_hat) * filt_rate*q_hat)
        # spectral enstrophy transfer of filter
        tzfilt = cp.real(cp.conj(q_hat) * filt_rate*q_hat)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of filter
        # In physical space using spectral aggregation
        tefilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tefilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of filter
        # In physical space 
        tzfilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tzfilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of filter
        fefilt_kk = np.cumsum(tefilt_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of filter
        fzfilt_kk = np.cumsum(tzfilt_kk[::-1])[::-1]
        return tefilt_kk, tzfilt_kk, fefilt_kk, fzfilt_kk 
    
    # Save and output methods
    def create_rst(self,nf):
        """Create NetCDF file for model state restart data
        
        Stores full vorticity field and time indices for RK4 or AB3 continuation
        
        Args:
            nf: File counter for restart checkpoint numbering
        """
        outdir = self.savedir
        # Create restart filename with counter
        if self.is_not_rst:
            nc_filename = os.path.join(outdir, "rst_%04d.nc"%(nf))
        else:
            nc_filename = os.path.join(outdir, "rst_r%04d.nc"%(nf))
        self.rstds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
        # Create dimensions for time, integration scheme index, and spatial grid
        time_dim = self.rstds.createDimension('time', None) 
        if self.ts_scheme  == 'rk4':
            ind_dim = self.rstds.createDimension('ind', 1) 
        elif self.ts_scheme ==  'ab3':
            ind_dim = self.rstds.createDimension('ind', 3) 
        x_dim = self.rstds.createDimension('x', self.Nx)
        y_dim = self.rstds.createDimension('y', self.Ny)
        # Create coordinate variables
        self.rst_times = self.rstds.createVariable('time', 'f8', ('time',))
        inds = self.rstds.createVariable('ind', 'f8', ('ind',))
        xs = self.rstds.createVariable('x', 'f8', ('x',))
        ys = self.rstds.createVariable('y', 'f8', ('y',))
        # Initialize time step indices based on integration scheme
        if self.ts_scheme  == 'rk4':
            inds[:] = np.array([0,])
        elif self.ts_scheme ==  'ab3':
            inds[:] = np.array([0,1,2])
        # Set spatial coordinates from GPU arrays
        xs[:] = self.x.get()
        ys[:] = self.y.get()

        # Create data variable for vorticity field
        self.qrst_var = self.rstds.createVariable('qrst', 'f8', ('time','ind', 'y', 'x'), zlib=False)

        # Store simulation parameters as global attributes
        self.rstds.description = "QG Turbulence Simulation RST file"
        self.rstds.dt = self.dt
        self.rstds.Nx = self.Nx
        self.rstds.Ny = self.Ny
        self.rstds.Lx = self.Lx
        self.rstds.Ly = self.Ly
        self.rstds.ts_scheme = self.ts_scheme
        self.rstds.kf = self.fscale
        self.rstds.friction = self.friction
        self.rstds.hyperorder=self.hyperorder
        self.rstds.visc2 = self.visc2
        self.rstds.hyvisc = self.hyvisc
        self.rstds.gamma = self.gamma
        self.rstds.beta = self.beta
        self.rstds.cl = self.cl

    def create_nc(self,nf):
        """Create NetCDF file for output diagnostics
        
        Initializes file with dimensions and coordinate variables for diagnostics output
        
        Args:
            nf: File counter for output numbering
        """
        outdir = self.savedir
        # Create output filename with counter
        if self.is_not_rst:
            nc_filename = os.path.join(outdir, "output_%04d.nc"%(nf))
        else:
            nc_filename = os.path.join(outdir, "output_r%04d.nc"%(nf))
        self.ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
        # Create dimensions for time, spatial grid, and wavenumber spectrum
        time_dim = self.ds.createDimension('time', None) 
        x_dim = self.ds.createDimension('x', self.Nx)
        y_dim = self.ds.createDimension('y', self.Ny)
        k_dim = self.ds.createDimension('k',len(self.kk_iso))
        # Create coordinate variables
        self.times = self.ds.createVariable('time', 'f8', ('time',))
        xs = self.ds.createVariable('x', 'f8', ('x',))
        ys = self.ds.createVariable('y', 'f8', ('y',))
        # Set spatial coordinates from GPU arrays
        xs[:] = self.x.get()
        ys[:] = self.y.get()
        # Create wavenumber coordinate in isotropic spectrum
        kk = self.ds.createVariable('k','f8',('k',))
        kk[:] = self.kk_iso.get()
        ## prognostic variable
        self.q_var = self.ds.createVariable('q', 'f8', ('time', 'y', 'x'), zlib=False)
        self.psi_var = self.ds.createVariable('psi', 'f8', ('time', 'y', 'x'), zlib=False)
        self.rv_var = self.ds.createVariable('rv', 'f8', ('time', 'y', 'x'), zlib=False)
        ## diagonistic variable
        ## invariant quantities
        self.Etot_var = self.ds.createVariable('Etot', 'f8', ('time',))
        self.Ztot_var = self.ds.createVariable('Ztot', 'f8', ('time',))
        self.Ek_var = self.ds.createVariable('Ek', 'f8', ('time', 'k'), zlib=False)
        self.Zk_var = self.ds.createVariable('Zk', 'f8', ('time', 'k'), zlib=False)
        ## tendency budget
        # non-linear advection
        self.tenlk_var = self.ds.createVariable('tenlk', 'f8', ('time', 'k'), zlib=False)
        self.tznlk_var = self.ds.createVariable('tznlk', 'f8', ('time', 'k'), zlib=False)
        # forcing 
        self.tefk_var = self.ds.createVariable('tefk', 'f8', ('time', 'k'), zlib=False)
        self.tzfk_var = self.ds.createVariable('tzfk', 'f8', ('time', 'k'), zlib=False)
        # friction
        self.tefrick_var = self.ds.createVariable('tefrick', 'f8', ('time', 'k'), zlib=False)
        self.tzfrick_var = self.ds.createVariable('tzfrick', 'f8', ('time', 'k'), zlib=False)
        # viscosity
        self.tevisck_var = self.ds.createVariable('tevisck', 'f8', ('time', 'k'), zlib=False)
        self.tzvisck_var = self.ds.createVariable('tzvisck', 'f8', ('time', 'k'), zlib=False)
        # filter
        self.tefiltk_var = self.ds.createVariable('tefiltk', 'f8', ('time', 'k'), zlib=False)
        self.tzfiltk_var = self.ds.createVariable('tzfiltk', 'f8', ('time', 'k'), zlib=False)
        ## flux budget
        # non-linear advection
        self.fenlk_var = self.ds.createVariable('fenlk', 'f8', ('time', 'k'), zlib=False)
        self.fznlk_var = self.ds.createVariable('fznlk', 'f8', ('time', 'k'), zlib=False)
        # forcing 
        self.fefk_var = self.ds.createVariable('fefk', 'f8', ('time', 'k'), zlib=False)
        self.fzfk_var = self.ds.createVariable('fzfk', 'f8', ('time', 'k'), zlib=False)
        # friction
        self.fefrick_var = self.ds.createVariable('fefrick', 'f8', ('time', 'k'), zlib=False)
        self.fzfrick_var = self.ds.createVariable('fzfrick', 'f8', ('time', 'k'), zlib=False)
        # viscosity
        self.fevisck_var = self.ds.createVariable('fevisck', 'f8', ('time', 'k'), zlib=False)
        self.fzvisck_var = self.ds.createVariable('fzvisck', 'f8', ('time', 'k'), zlib=False)
        # filter
        self.fefiltk_var = self.ds.createVariable('fefiltk', 'f8', ('time', 'k'), zlib=False)
        self.fzfiltk_var = self.ds.createVariable('fzfiltk', 'f8', ('time', 'k'), zlib=False)

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
        self.ds.visc2 = self.visc2
        self.ds.hyvisc = self.hyvisc
        self.ds.gamma = self.gamma
        self.ds.beta = self.beta
        self.ds.cl = self.cl

    def save_rst(self,it):
        """Save model state to restart file for integration continuation
        
        Stores vorticity and time derivatives required for RK4 or AB3 schemes
        
        Args:
            it: Output record number index
        """
        # Transform vorticity and time derivatives to physical space
        q_r = ifft2(self.q_hat).real.get()
        k1_p_r = ifft2(self.k1_p).real.get()
        k1_pp_r = ifft2(self.k1_pp).real.get()
        # Record simulation time
        self.rst_times[it] = self.t
        # Store time history based on integration scheme
        if self.ts_scheme == 'ab3':
            # AB3 requires 3 previous steps
            self.qrst_var[it,0,:,:] = k1_pp_r
            self.qrst_var[it,1,:,:] = k1_p_r
            self.qrst_var[it,2,:,:] = q_r
        elif self.ts_scheme == 'rk4':
            # RK4 only requires current state
            self.qrst_var[it,0,:,:] = q_r
        # Flush to disk
        self.rstds.sync()
        del q_r, k1_p_r, k1_pp_r
        gc.collect()


    def save_var(self,it):
        """Save diagnostic variables to output file
        
        Computes and stores spectral energy/enstrophy and all budget terms
        
        Args:
            it: Output record number index
        """
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
        # Energy and enstrophy total quantities
        self.Etot_var[it] = self.get_Etot(self.p_hat)
        self.Ztot_var[it] = self.get_Ztot(self.q_hat)
        self.Ek_var[it,:] = self.get_Ek(self.p_hat) 
        self.Zk_var[it,:] = self.get_Zk(self.q_hat) 
        # Energy and enstrophy budget terms
        # Non-linear advection transfer and flux
        self.tenlk_var[it,:],self.tznlk_var[it,:], self.fenlk_var[it,:], self.fznlk_var[it,:] = self.get_diagNL(self.p_hat,self.q_hat)
        # Forcing transfer and flux
        self.tefk_var[it,:], self.tzfk_var[it,:], self.fefk_var[it,:], self.fzfk_var[it,:] = self.get_diagF(self.p_hat,self.q_hat,self.force_q)
        # Friction dissipation and flux
        self.tefrick_var[it,:], self.tzfrick_var[it,:],self.fefrick_var[it,:], self.fzfrick_var[it,:] = self.get_diagFric(self.p_hat,self.q_hat)
        # Hyperviscosity dissipation and flux
        self.tevisck_var[it,:], self.tzvisck_var[it,:], self.fevisck_var[it,:], self.fzvisck_var[it,:] = self.get_diagVisc(self.p_hat,self.q_hat)
        # Spectral filter loss and flux
        self.tefiltk_var[it,:], self.tzfiltk_var[it,:],self.fefiltk_var[it,:], self.fzfiltk_var[it,:] = self.get_diagFilt(self.p_hat,self.q_hat)

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
        tecda,_,fecda,_ = self.get_diagCda(self.p_hat,self.q_hat,self.cda_term)
        # Normalize to mean per grid point
        norm_fac = 1.0 / (self.Nx * self.Ny)
        
        # Scale energy tendencies
        tenl *= norm_fac
        teF *= norm_fac
        tevisc *= norm_fac
        tefric *= norm_fac
        tefilt *= norm_fac
        tecda *= norm_fac
        
        # Scale energy fluxes
        fenl *= norm_fac
        feF *= norm_fac
        fevisc *= norm_fac
        fefric *= norm_fac
        fefilt *= norm_fac
        fecda *= norm_fac
        
        # Scale energy spectrum
        Ek = Ek * norm_fac

        # Compute residual budget terms
        # Tendency residual: should sum to zero in steady state
        te_sum = tenl + teF + tevisc + tefric + tefilt + tecda
        
        # Flux residual: cumulative sum of all fluxes
        fe_sum = fenl + feF + fevisc + fefric + fefilt + fecda

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
        ax_pv.set_title(f'Potential Vorticity (t={self.t:.2f})', fontsize=30, fontweight='bold')
        ax_pv.set_xlabel('x')
        ax_pv.set_ylabel('y')
        cbar = fig.colorbar(im, ax=ax_pv, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('PV')

        # Common Wavenumber Axis 
        ks = self.kk_iso.get() /(2*cp.pi/self.Lx)

        # Panel 2: Energy Spectrum
        ax_spec.loglog(ks, Ek, color='tab:blue', linewidth=3, label='Energy Spec')
        # References
        ks_direct = np.array([18., 80.]) /(2*cp.pi/self.Lx)
        ks_inv = np.array([5., 16.]) /(2*cp.pi/self.Lx)
        ax_spec.axvline(self.fscale/(2*cp.pi/self.Lx), color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax_spec.loglog(ks_direct, 0.5 * ks_direct**-3, 'k--', label='$k^{-3}$', alpha=0.6)
        ax_spec.loglog(ks_inv, 0.1 * ks_inv**-(5/3), 'k-.', label='$k^{-5/3}$', alpha=0.6)
        
        ax_spec.set_title('Isotropic KE Spectrum', fontweight='bold')
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
        ax_tendency.semilogx(ks, tecda, label='CDA', color='tab:red', linewidth=2.5)
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
        ax_flux.semilogx(ks, fecda, label='Cda. Flux', color='tab:red', linewidth=2.5)
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
    def run(self,scheme='ab3',tmax=40,tsave=200,nsave=100,savedir='run_0',saveplot=False):
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
        # Initialize or continue from restart time
        if self.trst:
            self.t = self.trst
        else:
            self.t = 0    
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        # Save restart files every 1 time unit
        tsrst = int(1/self.dt) 
        nrst = nsave
        insave=nsave
        inrst = nrst
        nf=0
        nfrst=0

        # Main time integration loop
        for n in range(int(tmax/self.dt)+1):
            self.n_steps = n
            # Check if need to create new output file
            if insave == nsave:
                itsave =0 
                if nf > 0 : self.ds.close()
                self.create_nc(nf)
                insave=0 
                nf+=1
            # Save diagnostics at specified intervals
            if n%tsave == 0:
                self.save_var(itsave)
                # Compute and print energy and diagnostic statistics
                E_crt = self.get_Etot(self.p_hat)/self.Nx/self.Ny
                Vrms_crt = self.get_Vrms(self.p_hat)
                Qrms_crt = self.get_Qrms(self.q_hat)
                print(f"   step {self.n_steps:7d}  t={self.t:9.6f}s E={E_crt:.4e} Vrms={Vrms_crt:.4e} Qrms={Qrms_crt:.4e}", end="\n")
                if saveplot:
                    self.plot_diag()
                itsave +=1
                insave +=1

            # Check if need to create new restart file
            if self.n_steps % tsrst==0:
                if inrst == nrst:
                    itrst =0 
                    if nfrst > 0 : self.rstds.close()
                    self.create_rst(nfrst)
                    inrst=0 
                    nfrst+=1

                self.save_rst(itrst)
                itrst +=1
                inrst +=1
            # Advance model state by one timestep
            self._step_forward()
            self.t += self.dt
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
