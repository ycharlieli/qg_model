
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
    # 2d turbulence model
    def __init__(self, Nx, Ny, Lx=2*cp.pi, Ly=2*cp.pi, dt=0.001,
                 beta=0, gamma=0, friction=0,visc2 = 0,hyperorder=1,sp_filtr=False,cl=0,forcing=None,wscale=4,winput=3):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.x = cp.linspace(0,self.Lx,self.Nx)
        self.y = cp.linspace(0,self.Ly,self.Ny)
        self._init_grid()
        # parameter of qg
        self.beta = beta
        self.gamma = gamma
        self.friction = friction # large scale friction 
        self.visc2 = visc2  #eddy viscosity
        self.hyvisc = 10*1/(self.Nx*(2*cp.pi/self.Lx))**(hyperorder*2) # viscosity
        self.hyperorder = hyperorder # order of hyper viscosity, 1-> Newnation 2-> biharmonic ...
        self.sp_filtr = sp_filtr # spectral filter impose on the tail of spectral (Arbic 2003)
        self.cl = cl #leith parameter
        self.forcing = forcing
        self.wscale = wscale # scale of wind
        self.winput = winput # enstrophy injection rate of wind
        self.force_q = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self._prebuild_operator()
        self._my_div()
        # gpu or cpu?  backend
###
    def _init_grid(self):
        self.x = cp.linspace(0,self.Lx,self.Nx)
        self.y = cp.linspace(0,self.Ly,self.Ny)
        self.x2d, self.y2d = cp.meshgrid(self.x,self.y)
        # Grid indices
        nx = cp.arange(self.Nx); nx[int(self.Nx/2):] -= self.Nx # or np.fft.fftfreq(Nx)*Nx
        ny = cp.arange(self.Ny); ny[int(self.Ny/2):] -= self.Ny
        self.nx2d, self.ny2d = cp.meshgrid(nx,ny)
        kx = 2*cp.pi*nx/self.Lx
        ky = 2*cp.pi*ny/self.Ly
        # get the 2d wavenumber
        self.kx2d, self.ky2d = cp.meshgrid(kx,ky) 
        # get K, i.e. 2d isotropic wave number 
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
        p_hat = self.inversion*q_hat
        rv_hat = q_hat + self.gamma**2*p_hat
        jacobian_term = self.compute_jacobian(p_hat,q_hat)
        beta_term = self.beta*self.kx2d*1j*p_hat
        damping_term = (-self.friction + self.visc2*self.lap+ self.hylap)*rv_hat
        damping_term+=self.compute_leith_term()
        
        return -jacobian_term-beta_term+damping_term+self.force_q

    def _step_forward(self):
        # based on rk4
        if self.forcing == 'wind':
            self._set_windforce()
        elif self.forcing =='thuburn':
            self.force_q = fft2(0.1*cp.sin(32*np.pi*self.x2d))
        q = self.q_hat

        k1 = self._get_rhs(q)
        k2 = self._get_rhs(q + 0.5 * self.dt * k1)
        k3 = self._get_rhs(q + 0.5 * self.dt * k2)
        k4 = self._get_rhs(q + self.dt * k3)
        
        self.q_hat = q + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.q_hat *=self.filtr
        self.p_hat = self.inversion*self.q_hat
        self.rv_hat = self.q_hat + self.gamma**2*self.p_hat
        
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
        self.rv_hat = self.lap*self.p_hat
        self.q_hat = self.rv_hat - self.gamma**2*self.p_hat
        
    def set_initial_condition(self,scheme='jcm1984',k_peak=6,krange=[3,5],eini=0,ls=3,ss=-3,q_ini=None,):
        self.eini = eini
        self.k_peak=k_peak
        if scheme == 'jcm1984':
            # kk**(-A)*(1 + (kk/k0)**4)**(-B)
            # generate Fourier conponent of initial streamfunction field
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
            rand_p = fft2(cp.random.randn(*self.kk.shape))
            rand_p[self.kk < krange[0]] = 0.0
            rand_p[self.kk > krange[1]] = 0.0
            rand_p[0,0] = 0.0
            self.p_hat = rand_p.copy()
            if eini:
                self._norm_energy()
        elif scheme == 'manual':
            # give initial q field manually
            self.q_hat = q_ini.copy()
            self.p_hat = self.inversion*self.q_hat
            self.rv_hat= self.lap*self.p_hat
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
    # def upscale(self,Nxup,Nyup):

    def _set_windforce(self):
        # graham 2013 and Frezat 2022
        # only valid for 2pi domain
        phi_x = cp.pi*cp.sin(1.5*self.t)
        phi_y = cp.pi*cp.sin(1.4*self.t)
        Fq = cp.cos(self.wscale*self.y2d + phi_y) - cp.cos(self.wscale*self.x2d + phi_x) 
        Fq_hat = fft2(Fq)
        inputF = cp.sum(cp.real(cp.conj(self.q_hat)*Fq_hat))/(self.Nx*self.Ny)**2 # current enstrophy injection
        norm_fac = 1.*(self.winput)/inputF
        Fq_hat *= norm_fac
        self.force_q = Fq_hat
        # plt.imshow(ifft2(Fq_hat).real.get())
        # plt.colorbar()
        # plt.show()
    
## diagnostic term
    def get_Ek(self,p_hat):
        # Isotropic Energy spectrum
        # Energy density in spectral space
        ene_dens = 0.5*self.kk**2*cp.abs(p_hat)**2
        # Physical space using Parseval's Theorem
        norm_fac = 1/(self.Nx*self.Ny)
        ene_kk = npg.aggregate(self.kk_idx.ravel().get(),ene_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ene_kk
    def get_Etot(self,p_hat):
        # Total Energy
        ene_kk = self.get_Ek(p_hat)
        ene_tot = np.sum(ene_kk) 
        return ene_tot
    def get_Zk(self,q_hat):
        # Enstrophy density in spectral space
        ens_dens = 0.5*np.abs(q_hat)**2
        norm_fac = 1/(self.Nx*self.Ny)
        # isotropic enstrophy spectrum in physical space using Parseval's Theorem
        ens_kk = npg.aggregate(self.kk_idx.ravel().get(),ens_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        return ens_kk
    def get_Ztot(self,q_hat):
        # Total Enstrophy
        ens_kk = self.get_Zk(q_hat)
        ens_tot = np.sum(ens_kk)
        return ens_tot
    def get_TENL(self,p_hat,q_hat):
        jacobian_term = self.compute_jacobian(p_hat,q_hat)
        # spectral energy transfer of non-linear advection
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        return tenl
    
    def get_TZNL(self,p_hat,q_hat):
        jacobian_term = self.compute_jacobian(p_hat,q_hat)
        # spectral enstrophy transfer of non-linear advection
        tznl = -cp.real(cp.conj(q_hat)*jacobian_term)
        return tznl

    def get_diagNL(self,p_hat,q_hat):
        jacobian_term = self.compute_jacobian(p_hat,q_hat)
        # spectral energy transfer of non-linear advection
        tenl = cp.real(cp.conj(p_hat)*jacobian_term)
        # spectral enstrophy transfer of non-linear advection
        tznl = -cp.real(cp.conj(q_hat)*jacobian_term)

        # Isotropic spectrum of energy transfer of non-linear advection 
        # In physical space 
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
        # spectral energy transfer of forcing
        teF = -cp.real(cp.conj(p_hat)*force_q)
        # spectral enstrophy transfoer of forcing
        tzF = cp.real(cp.conj(q_hat)*force_q)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of non-linear advection
        # In physical space 
        teF_kk = npg.aggregate(self.kk_idx.ravel().get(),teF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of non-linear advection
        # In physical space 
        tzF_kk =npg.aggregate(self.kk_idx.ravel().get(),tzF.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of forcing
        feF_kk = np.cumsum(teF_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of forcing
        fzF_kk = np.cumsum(tzF_kk[::-1])[::-1]

        return teF_kk, tzF_kk, feF_kk, fzF_kk
    
    def get_diagFric(self,p_hat,q_hat):
        # spectral energy transfer of friction
        tefric = -2*self.friction* 0.5*self.kk**2*cp.abs(p_hat)**2
        # spectral enstrophy transfer of friction
        tzfric = -2*self.friction* 0.5*cp.abs(q_hat)**2
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of friction
        # In physical space 
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
        # spectral energy transfer of viscosity
        tevisc = -cp.real(cp.conj(p_hat)*(self.visc2*self.lap+self.hylap)*q_hat) #TODO add parameterized term (Leith)
        # spectral enstrophy transfer of viscosity
        tzvisc = cp.real(cp.conj(q_hat)*(self.visc2*self.lap+self.hylap)*q_hat)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of viscosity
        # In physical space 
        tevisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tevisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of friction
        # In physical space 
        tzvisc_kk = npg.aggregate(self.kk_idx.ravel().get(),tzvisc.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of friction
        fevisc_kk = np.cumsum(tevisc_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of friction
        fzvisc_kk = np.cumsum(tzvisc_kk[::-1])[::-1]
        return tevisc_kk, tzvisc_kk, fevisc_kk, fzvisc_kk

    def get_diagFilt(self,p_hat,q_hat):
        filt_rate = (self.filtr - 1.) / self.dt
        tefilt = -cp.real(cp.conj(p_hat) * filt_rate*q_hat)
        tzfilt = cp.real(cp.conj(q_hat) * filt_rate*q_hat)
        norm_fac = 1/(self.Nx*self.Ny)
        # Isotropic spectrum of energy transfer of friction
        # In physical space 
        tefilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tefilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of enstrophy transfer of friction
        # In physical space 
        tzfilt_kk = npg.aggregate(self.kk_idx.ravel().get(),tzfilt.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # Isotropic spectrum of energy flux of friction
        fefilt_kk = np.cumsum(tefilt_kk[::-1])[::-1]
        # Isotropic spectrum of enstrophy flux of friction
        fzfilt_kk = np.cumsum(tzfilt_kk[::-1])[::-1]
        return tefilt_kk, tzfilt_kk, fefilt_kk, fzfilt_kk
    
    
    
###   
    def create_nc(self,nf):
        outdir = self.savedir
        nc_filename = os.path.join(outdir, "output_%04d.nc"%(nf))
        self.ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')
        time_dim = self.ds.createDimension('time', None) 
        x_dim = self.ds.createDimension('x', self.Nx)
        y_dim = self.ds.createDimension('y', self.Ny)
        k_dim = self.ds.createDimension('k',len(self.kk_iso))
        self.times = self.ds.createVariable('time', 'f8', ('time',))
        xs = self.ds.createVariable('x', 'f8', ('x',))
        ys = self.ds.createVariable('y', 'f8', ('y',))
        xs[:] = self.x.get()
        ys[:] = self.y.get()
        kk = self.ds.createVariable('k','f8',('k',))
        kk[:] = self.kk_iso.get()
        ## prognostic variable
        self.q_var = self.ds.createVariable('q', 'f8', ('time', 'x', 'y'), zlib=False)
        self.psi_var = self.ds.createVariable('psi', 'f8', ('time', 'x', 'y'), zlib=False)
        self.rv_var = self.ds.createVariable('rv', 'f8', ('time', 'x', 'y'), zlib=False)
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
        self.ds.kf = self.wscale
        self.ds.k0 = self.k_peak
        self.ds.friction = self.friction
        self.ds.hyvisc = self.hyvisc
        self.ds.gamma = self.gamma
        self.ds.beta = self.beta
        self.ds.cl = self.cl


    def save_var(self,it):
        p_r = ifft2(self.p_hat).real.get()
        q_r = ifft2(self.q_hat).real.get()
        rv_r = ifft2(self.rv_hat).real.get()
        self.times[it] = self.t
        ## prognostic variable
        self.q_var[it,:,:] = q_r
        self.psi_var[it,:,:] = p_r
        self.rv_var[it,:,:] = rv_r
        # diagnostic variable
        # invariant quantities
        self.Etot_var[it] = self.get_Etot(self.p_hat)
        self.Ztot_var[it] = self.get_Ztot(self.q_hat)
        self.Ek_var[it,:] = self.get_Ek(self.p_hat) 
        self.Zk_var[it,:] = self.get_Zk(self.q_hat) 
        # tendency budget
        self.tenlk_var[it,:],self.tznlk_var[it,:], self.fenlk_var[it,:], self.fznlk_var[it,:] = self.get_diagNL(self.p_hat,self.q_hat)
        # forcing 
        self.tefk_var[it,:], self.tzfk_var[it,:], self.fefk_var[it,:], self.fzfk_var[it,:] = self.get_diagF(self.p_hat,self.q_hat,self.force_q)
        # friction 
        self.tefrick_var[it,:], self.tzfrick_var[it,:],self.fefrick_var[it,:], self.fzfrick_var[it,:] = self.get_diagFric(self.p_hat,self.q_hat)
        # viscosity
        self.tevisck_var[it,:], self.tzvisck_var[it,:], self.fevisck_var[it,:], self.fzvisck_var[it,:] = self.get_diagVisc(self.p_hat,self.q_hat)
        # filter
        self.tefiltk_var[it,:], self.tzfiltk_var[it,:],self.fefiltk_var[it,:], self.fzfiltk_var[it,:] = self.get_diagFilt(self.p_hat,self.q_hat)

        self.ds.sync()
        del q_r, p_r, rv_r
        gc.collect()
        
    def plot_diag(self, save_path=None):
        """
        Plots the current state and Energy budgets (Tendency and Flux).
        Layout:
          - Top (4x4): PV Field
          - Bottom Left: Energy Spectrum
          - Bottom Center: Energy Tendency Budget
          - Bottom Right: Energy Flux Budget
        """
        # --- 1. Compute Diagnostics ---
        Ek = self.get_Ek(self.p_hat)
        
        # A. Non-Linear Transfer & Flux
        tenl, _, fenl, _ = self.get_diagNL(self.p_hat, self.q_hat)
        
        # B. Forcing (Injection) & Flux
        teF, _, feF, _ = self.get_diagF(self.p_hat, self.q_hat, self.force_q)
        
        # C. Dissipation Terms (Calculated Separately)
        tevisc, _, fevisc, _ = self.get_diagVisc(self.p_hat, self.q_hat)
        tefric, _, fefric, _ = self.get_diagFric(self.p_hat, self.q_hat)
        tefilt, _, fefilt, _ = self.get_diagFilt(self.p_hat, self.q_hat)
        # --- 2. Normalize to Mean (Density) ---
        # Current variables are "Total Sums". Divide by grid size to get "Mean per point".
        norm_fac = 1.0 / (self.Nx * self.Ny)
        
        # Scale Tendencies
        tenl *= norm_fac
        teF *= norm_fac
        tevisc *= norm_fac
        tefric *= norm_fac
        tefilt *= norm_fac
        
        # Scale Fluxes
        fenl *= norm_fac
        feF *= norm_fac
        fevisc *= norm_fac
        fefric *= norm_fac
        fefilt *= norm_fac
        
        # Scale Energy Spectrum
        Ek = Ek * norm_fac

        # D. Residuals (Should be ~0 in steady state)
        # Tendency Residual: dE/dt = NL + Forcing + Viscosity + Friction
        te_sum = tenl + teF + tevisc + tefric + tefilt
        
        # Flux Residual: Sum of cumulative fluxes
        fe_sum = fenl + feF + fevisc + fefric + fefilt

        # --- 3. Setup Figure ---
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(30, 18), tight_layout=True)
        gs = fig.add_gridspec(6, 6)

        # Assign Axes
        ax_pv = fig.add_subplot(gs[0:4, 1:5])
        ax_spec = fig.add_subplot(gs[4:, 0:2])
        ax_tendency = fig.add_subplot(gs[4:, 2:4]) # Center Bottom
        ax_flux = fig.add_subplot(gs[4:, 4:])      # Right Bottom

        # --- 4. Plotting ---

        # Panel 1: PV Field
        q_phys = ifft2(self.q_hat).real.get()
        im = ax_pv.imshow(q_phys, cmap=self.my_div, 
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
        ax_spec.axvline(self.wscale/(2*cp.pi/self.Lx), color='k', linestyle='--', linewidth=1.5, alpha=0.5)
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
        ax_tendency.semilogx(ks, tefric, label='Friction', color='tab:red', linewidth=2.5)
        ax_tendency.semilogx(ks, tefilt, label='Filter', color='tab:purple', linewidth=2.5)
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
        ax_flux.semilogx(ks, fefric, label='Fric. Flux', color='tab:red', linewidth=2.5)
        ax_flux.semilogx(ks, fefilt, label='Filt. Flux', color='tab:purple', linewidth=2.5)
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
        """Called inside run loop to save the diagnostic figure."""
        # Create directory
        outdir = os.path.join(self.savedir, "figs")
        os.makedirs(outdir, exist_ok=True)
        
        # Save figure with time-stamp
        filename = os.path.join(outdir, f"snap_{nstep:04d}.png")
        self.plot_diag(save_path=filename)

    def compute_jacobian(self,p_hat,q_hat):
        dxp_hat = self.kx2d*1j*p_hat
        dyp_hat = self.ky2d*1j*p_hat
        dxq_hat = self.kx2d*1j*q_hat
        dyq_hat = self.ky2d*1j*q_hat
        # zeropadding to subsample with the high frequency to avoid aliasing 
        dxq_r = ifft2(self._padding(dxq_hat)).real
        dyq_r = ifft2(self._padding(dyq_hat)).real
        dxp_r = ifft2(self._padding(dxp_hat)).real
        dyp_r = ifft2(self._padding(dyp_hat)).real
    
        jacob_r = dyq_r*dxp_r-dyp_r*dxq_r
    
        jacob_hat = self._unpadding(fft2(jacob_r))
    
        return jacob_hat

    def compute_leith_term(self):
        dxrv_hat = self.kx2d*1j*self.rv_hat
        dyrv_hat = self.ky2d*1j*self.rv_hat
        
        dxrv_r = ifft2(self._padding(dxrv_hat)).real
        dyrv_r = ifft2(self._padding(dyrv_hat)).real
        
        grad_rv_r = cp.sqrt(dxrv_r**2+dyrv_r**2)
        
        nabla = self.Lx/self.Nx
        nu_e = (self.cl*nabla)**3*grad_rv_r
        
        flux_x_r = nu_e*dxrv_r
        flux_y_r = nu_e*dyrv_r

        flux_x_hat = self.kx2d*1j*self._unpadding(fft2(flux_x_r))
        flux_y_hat = self.ky2d*1j*self._unpadding(fft2(flux_y_r))

        

        return flux_x_hat+flux_y_hat
        
    def run(self,tmax=40,tsave=200,nsave=100,tplot=1000,savedir='run_0',saveplot=False):
        self.tmax = tmax
        self.tsave = tsave
        self.t = 0     
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        insave=nsave
        inplot = 0
        nf=0

        for n in range(int(tmax/self.dt)+1):
            if insave == nsave:
                itsave =0 #time index
                if nf > 0 : self.ds.close()
                self.create_nc(nf)
                insave=0 #save number index
                nf+=1
            if n%tsave == 0:
                self.save_var(itsave)
                print(f"step {n:7d}  t = {self.t:9.6f} s  E = {self.get_Etot(self.p_hat)/self.Nx/self.Ny:.4e}", end="\n")
                if saveplot:
                    self.plot_diag()
                itsave +=1
                insave +=1
            if n%tplot ==0:
                print(f"\t saving figure", end="\n")
                self.save_snapshot(inplot) 
                inplot+=1
            
            self._step_forward()
            self.t += self.dt
        self.ds.close()
        print('Done.')
    
    # def cda_run(self,truth,...)
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
