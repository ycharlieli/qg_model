# %%
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
                 beta=0, gamma=0, friction=0,visc=0,hyperorder=1,sp_filtr=False,cl=0,forcing=None,wscale=4):
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
        self.visc = visc # viscosity
        self.hyperorder = hyperorder # order of hyper viscosity, 1-> Newnation 2-> biharmonic ...
        self.sp_filtr = sp_filtr # spectral filter impose on the tail of spectral (Arbic 2003)
        self.cl = cl #leith parameter
        self.forcing = forcing
        self.wscale = wscale # scale of wind
        self.force_q = cp.zeros((self.Nx, self.Ny), dtype=np.complex128)
        self._prebuild_operator()
        self._my_div()
        # gpu or cpu?  backend

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
        self.kk_idx = cp.round(cp.sqrt(self.nx2d**2+self.ny2d**2)).astype('int')
        self.kk_idx_set = cp.unique(self.kk_idx)
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
        damping_term = (-self.friction + (-1)**(self.hyperorder+1)*self.visc*(self.lap**(self.hyperorder)))*rv_hat
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
        self.ene_kk, self.ens_kk, self.eneflux_kk, self.ensflux_kk, self.ene_tot, self.ens_tot = self.get_diag(self.p_hat,self.q_hat)
        norm_fac = cp.sqrt(0.5/(self.ene_tot/(self.Nx*self.Ny)))
        self.p_hat = norm_fac*self.p_hat
        # initial relavitve vorticity (rv)
        self.rv_hat = self.lap*self.p_hat
        # initial potential vorticity (q)
        self.q_hat = self.rv_hat - self.gamma**2*self.p_hat
        self.ene_kk, self.ens_kk, self.eneflux_kk, self.ensflux_kk, self.ene_tot, self.ens_tot = self.get_diag(self.p_hat,self.q_hat)
        self.rv_hat = self.lap*self.p_hat
        self.q_hat = self.rv_hat - self.gamma**2*self.p_hat
        
    def set_initial_condition(self,scheme='jcm1984',k_peak=6,krange=[3,5],ls=3,ss=-3,q_ini=None):
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
            self._norm_energy()
        elif scheme == 'manual':
            # give initial q field manually
            self.q_hat = q_ini.copy()
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
        self.ene_kk, self.ens_kk, self.eneflux_kk, self.ensflux_kk, self.ene_tot, self.ens_tot = self.get_diag(self.p_hat,self.q_hat)
    # def upscale(self,Nxup,Nyup):
    
    def _set_windforce(self):
        # graham 2013 and Frezat 2022
        phi_x = cp.pi*cp.sin(1.5*self.t)
        phi_y = cp.pi*cp.sin(1.4*self.t)
        Fq = cp.cos(self.wscale*self.y2d + phi_y) - cp.cos(self.wscale*self.x2d + phi_x) 
        Fq_hat = fft2(Fq)
        Fq_hat[self.kk < 3.0] = 0.0
        Fq_hat[self.kk > 5.0] = 0.0
        Fq_hat[0, 0] = 0.0  # Zero mean
        Fens_tot = 0.5 * cp.sum(cp.abs(Fq_hat)**2) / (self.Nx**2 * self.Ny**2)
        norm_fac = cp.sqrt(3/Fens_tot)
        Fq_hat *= norm_fac
        self.force_q = Fq_hat
        # plt.imshow(ifft2(Fq_hat).real.get())
        # plt.colorbar()
        # plt.show()
        

    def get_diag(self,p_hat,q_hat):
        jacobian_term = self.compute_jacobian(p_hat,q_hat)
        # KE density in spectral space
        ene_dens = 0.5*self.kk**2*cp.abs(p_hat)**2
        # Enstrophy density in spectral space
        ens_dens = 0.5*np.abs(q_hat)**2
        # KE flux tendency in spectral space
        ene_flux = cp.real(cp.conj(p_hat)*jacobian_term)
        # Enstrophy flux tendency in spectral space
        ens_flux = -cp.real(cp.conj(q_hat)*jacobian_term)
        # isotropic energy spectrum in physical space using Parseval's Theorem
        norm_fac = 1/(self.Nx*self.Ny)
        ene_kk = npg.aggregate(self.kk_idx.ravel().get(),ene_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        ens_kk = npg.aggregate(self.kk_idx.ravel().get(),ens_dens.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # isotropic kinetic energy flux(tendency actually here) in physical space 
        eneflux_kk = npg.aggregate(self.kk_idx.ravel().get(),ene_flux.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # isotropic enstrophy flux (tendency actually here) in physical space
        ensflux_kk = npg.aggregate(self.kk_idx.ravel().get(),ens_flux.ravel().get(),func='sum')[self.kk_range.get()] * norm_fac
        # total energy in physical space using Parseval's Theorem
        ene_tot = np.sum(ene_kk) 
        ens_tot = np.sum(ens_kk) 
        
        return ene_kk, ens_kk, eneflux_kk, ensflux_kk, ene_tot, ens_tot
        
    def create_nc(self,nf,savedir):
        outdir = "./%s"%(savedir)
        os.makedirs(outdir, exist_ok=True)
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
        self.q_var = self.ds.createVariable('q', 'f8', ('time', 'x', 'y'), zlib=False)
        self.psi_var = self.ds.createVariable('psi', 'f8', ('time', 'x', 'y'), zlib=False)
        self.rv_var = self.ds.createVariable('rv', 'f8', ('time', 'x', 'y'), zlib=False)
        self.ene_var = self.ds.createVariable('energy', 'f8', ('time',))
        self.ens_var = self.ds.createVariable('enstrophy', 'f8', ('time',))
        self.enespec_var = self.ds.createVariable('ene_spec', 'f8', ('time', 'k'), zlib=False)
        self.ensspec_var = self.ds.createVariable('ens_spec', 'f8', ('time', 'k'), zlib=False)
        self.eneflux_var = self.ds.createVariable('ene_flux', 'f8', ('time', 'k'), zlib=False)
        self.ensflux_var = self.ds.createVariable('ens_flux', 'f8', ('time', 'k'), zlib=False)
        self.ds.description = "QG Turbulence Simulation"
        self.ds.dt = self.dt
        self.ds.Nx = self.Nx
        self.ds.Ny = self.Ny
        self.ds.k0 = self.k_peak
        self.ds.friction = self.friction
        self.ds.visc = self.visc
        self.ds.gamma = self.gamma
        self.ds.beta = self.beta
        self.ds.cl = self.cl


    def save_var(self,it):
        self.ene_kk, self.ens_kk, self.eneflux_kk, self.ensflux_kk, self.ene_tot, self.ens_tot = self.get_diag(self.p_hat,self.q_hat)
        p_r = ifft2(self.p_hat).real.get()
        q_r = ifft2(self.q_hat).real.get()
        rv_r = ifft2(self.rv_hat).real.get()
        self.times[it] = self.t
        self.q_var[it,:,:] = q_r
        self.psi_var[it,:,:] = p_r
        self.rv_var[it,:,:] = rv_r
        self.ene_var[it] = self.ene_tot
        self.ens_var[it] = self.ens_tot
        self.enespec_var[it,:] = self.ene_kk
        self.ensspec_var[it,:] = self.ens_kk
        self.eneflux_var[it,:] = self.eneflux_kk
        self.ensflux_var[it,:] = self.ensflux_kk
        self.ds.sync()
        
    def plot_diag(self):
        fig,axs = plt.subplots(1,5, figsize=(25,5),tight_layout=False)
        cq=axs[0].imshow(ifft2(self.q_hat).real.get(),cmap=my_div,vmin=-40,vmax=40)
        fig.colorbar(cq,ax=axs[0])
        axs[0].set_title('$\zeta-\gamma^2\psi$')
        cvor=axs[1].imshow(ifft2(self.rv_hat).real.get(),cmap=my_div,vmin=-40,vmax=40)
        fig.colorbar(cvor,ax=axs[1])
        axs[1].set_title('$\zeta$')
        cp=axs[2].imshow(ifft2(self.p_hat).real.get(),cmap=my_div,vmin=-1,vmax=1)
        fig.colorbar(cp,ax=axs[2])
        fig.suptitle('time=%15.5fsec'%t)
        axs[2].set_title('$\psi$')
        
        axs[3].loglog(self.kk_iso.get(),self.ene_kk)
        ks = np.array([3.,80])
        es = 1e3*ks**-3
        axs[3].loglog(ks,es,'k--')
        es = 10*ks**-5
        axs[3].loglog(ks,es,'k--')
        axs[3].set_xlim([1,int(Nx/2)])
        axs[3].set_ylim([1e-8,1])
        axs[3].set_title('Energy Spectrum')

        axs[4].plot(self.kk_iso.get(),self.eneflux_kk)
        axs[4].set_xlim([1,int(Nx/2)])
        axs[4].set_xscale('log')
        axs[4].set_ylim([-2*1e-1,2*1e-1])
        axs[4].set_title('KE flux')
        
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
        
    def run(self,tmax=40,tsave=200,nsave=100,savedir='run_0',tsnapshot=1000):
        self.tmax = tmax
        self.tsave = tsave
        self.tsnapshot = tsnapshot
        self.t = 0     
        insave=nsave
        nf=0

        for n in range(int(tmax/self.dt)+1):
            if insave == nsave:
                itsave =0 #time index
                if nf > 0 : self.ds.close()
                self.create_nc(nf,savedir)
                insave=0 #save number index
                nf+=1
            if n%tsave == 0:
                self.save_var(itsave)
                print(f"step {n:7d}  t = {self.t:9.6f} s  E = {self.ene_tot:.4e}", end="\n")
                itsave +=1
                insave +=1 
            # if n%tsnapshot == 0:
            #     self.save_snapshot()
                

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
