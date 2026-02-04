import numpy as np
import matplotlib.pyplot as plt
import dask
from matplotlib.colors import LinearSegmentedColormap

def get_my_div_cmap():
    my_div_color = np.array([
        [0,0,123], [9,32,154], [22,58,179], [34,84,204], [47,109,230],
        [63,135,247], [95,160,248], [137,186,249], [182,213,251], [228,240,254],
        [255,255,255], [250,224,224], [242,164,162], [237,117,113], [235,76,67],
        [233,52,37], [212,45,31], [188,39,26], [164,33,21], [140,26,17], [117,20,12]
    ]) / 255
    return LinearSegmentedColormap.from_list('div', my_div_color, N=256)

plt.rcParams.update({'font.size': 14})

def plot_pv_snap(ds, time_idx=-1,valmin=None,valmax=None):
    """Plots PV field from xarray dataset."""
    # Extract data
    q_phys = ds.q.sel(time=time_idx,method='nearest').values
    t_val = ds.time.sel(time=time_idx,method='nearest').values
    if not valmax:
        qmax = q_phys.max().round()
        valmax = (qmax - qmax%10)*0.8
    # Grid info
    Lx = ds.x.max().values # Assuming x starts at 0 and goes to Lx
    Ly = ds.y.max().values
    
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    
    im = ax.imshow(q_phys, cmap=get_my_div_cmap(),vmin=-valmax,vmax=valmax,
                   extent=[0, ds.Lx, ds.Ly,0])
    
    ax.set_title(f'Potential Vorticity (t={t_val:.2f})', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('PV')
    plt.show()

def plot_espec_snap(ds,time_idx=-1,valmin=None,valmax=None):
    """Plots Energy Spectrum from xarray dataset."""
    # Parameters
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny) # plot_diag applies this EXTRA normalization
    
    # Data
    Ek = ds.Ek.sel(time=time_idx,method='nearest').values * norm_fac
    t_val = ds.time.sel(time=time_idx,method='nearest').values
    k_iso = ds.k.values
    if not valmax:
        valmin = 1e-20*2*Ek.max()
        valmax = 2*Ek.max()
    # Scale wavenumbers (k / (2pi/Lx))
    ks = k_iso[:int(len(k_iso)*0.8)] / (2 * np.pi / Lx) 
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Plot Spectrum
    ax.loglog(ks, Ek[:int(len(k_iso)*0.8)], color='tab:blue', linewidth=3, label='Energy Spec')
    
    # References lines (k^-3 and k^-5/3)
    ks_direct = np.array([fscale+1., fscale*5.]) 
    ks_inv = np.array([fscale/5., fscale-1.])
    ax.axvline(fscale, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
  
    ax.loglog(ks_inv, 0.5 * ks_inv**-(5/3), 'k-.', label='$k^{-5/3}$', alpha=0.6)
    ax.loglog(ks_direct, 10 * ks_direct**-3, 'k--', label='$k^{-3}$', alpha=0.6)
    
    # Wavenumber of forcing (wscale not in attrs, infer or omit)
    # ax.axvline(...) 

    ax.set_title(f'Isotropic Energy Spectrum (t={t_val:.2f})', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('$E(k)$')
    ax.set_xlim([1, int(Nx/2)])
    ax.set_ylim([valmin,valmax])
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')
    plt.show()

def plot_espec_avg(ds,tmin=None,tmax=None,valmin=None,valmax=None,versus=None):
    """Plots Energy Spectrum from xarray dataset."""
    # Parameters
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny) # plot_diag applies this EXTRA normalization
    
    # Data
    Ek = ds.Ek.sel(time=slice(tmin,tmax)).values.mean(axis=0) * norm_fac
    if versus: Ek_ref = versus.Ek.sel(time=slice(tmin,tmax)).values.mean(axis=0) * norm_fac
    k_iso = ds.k.values
    if not valmax:
        valmin = 1e-20*2*Ek.max()
        valmax = 2*Ek.max()
    # Scale wavenumbers (k / (2pi/Lx))
    ks = k_iso[:int(len(k_iso)*0.8)] / (2 * np.pi / Lx) 
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Plot Spectrum
    ax.loglog(ks, Ek[:int(len(k_iso)*0.8)], color='tab:blue', linewidth=3, label='Energy Spec')
    if versus: ax.loglog(ks, Ek_ref[:int(len(k_iso)*0.8)], color='tab:red', linewidth=3, label='Energy Spec REF')
    
    # References lines (k^-3 and k^-5/3)
    ks_direct = np.array([fscale+1., fscale*5.]) 
    ks_inv = np.array([fscale/5., fscale-1.])
    ax.axvline(fscale, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
  
    ax.loglog(ks_inv, 0.5 * ks_inv**-(5/3), 'k-.', label='$k^{-5/3}$', alpha=0.6)
    ax.loglog(ks_direct, 10 * ks_direct**-3, 'k--', label='$k^{-3}$', alpha=0.6)
    
    # Wavenumber of forcing (wscale not in attrs, infer or omit)
    # ax.axvline(...) 

    ax.set_title('Isotropic Energy Spectrum', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('$E(k)$')
    ax.set_xlim([1, int(Nx/2)])
    ax.set_ylim([valmin,valmax])
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')
    plt.show()

def plot_tebudget_snap(ds, time_idx=-1):
    """Plots Energy Tendency Budget from xarray dataset."""
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny)
    
    # Select time slice
    ds_t = ds.sel(time=time_idx,method='nearest')
    t_val = ds.time.sel(time=time_idx,method='nearest').values
    
    # Load and Normalize terms
    tenl = ds_t.tenlk.values * norm_fac
    teF = ds_t.tefk.values * norm_fac
    tevisc = ds_t.tevisck.values * norm_fac
    tefric = ds_t.tefrick.values * norm_fac
    tefilt = ds_t.tefiltk.values * norm_fac
    
    # Calculate Residual
    te_sum = tenl + teF + tevisc + tefric + tefilt
    
    # Wavenumbers
    ks = ds.k.values / (2 * np.pi / Lx)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.semilogx(ks, tenl, label='NL Transfer', color='tab:blue', linewidth=2.5)
    ax.semilogx(ks, teF, label='Forcing', color='tab:green', linewidth=2.5)
    ax.semilogx(ks, tevisc, label='Viscosity', color='tab:orange', linewidth=2.5)
    ax.semilogx(ks, tefric, label='Friction', color='tab:red', linewidth=2.5)
    ax.semilogx(ks, tefilt, label='Filter', color='tab:purple', linewidth=2.5)
    ax.semilogx(ks, te_sum, 'k--', label='Sum (Residual)', linewidth=1.5)
    ax.axvline(fscale, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(0, color='k', linestyle='-', linewidth=1.5)
    ax.set_xlim([1, int(Nx/2)])
    ax.set_title(f'Energy Tendency Budget ($dE/dt$) (t={t_val:.2f})', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Rate')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize='small', loc='best')
    plt.show()

def plot_tebudget_avg(ds, tmin=None,tmax=None):
    """Plots Energy Tendency Budget from xarray dataset."""
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny)
    
    # Select time slice
    ds_t = ds.sel(time=slice(tmin,tmax))
    
    # Load and Normalize terms
    tenl = ds_t.tenlk.mean(axis=0).values * norm_fac
    teF = ds_t.tefk.mean(axis=0).values * norm_fac
    tevisc = ds_t.tevisck.mean(axis=0).values * norm_fac
    tefric = ds_t.tefrick.mean(axis=0).values * norm_fac
    tefilt = ds_t.tefiltk.mean(axis=0).values * norm_fac
    
    # Calculate Residual
    te_sum = tenl + teF + tevisc + tefric + tefilt
    
    # Wavenumbers
    ks = ds.k.values / (2 * np.pi / Lx)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.semilogx(ks, tenl, label='NL Transfer', color='tab:blue', linewidth=2.5)
    ax.semilogx(ks, teF, label='Forcing', color='tab:green', linewidth=2.5)
    ax.semilogx(ks, tevisc, label='Viscosity', color='tab:orange', linewidth=2.5)
    ax.semilogx(ks, tefric, label='Friction', color='tab:red', linewidth=2.5)
    ax.semilogx(ks, tefilt, label='Filter', color='tab:purple', linewidth=2.5)
    ax.semilogx(ks, te_sum, 'k--', label='Sum (Residual)', linewidth=1.5)
    ax.axvline(fscale, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(0, color='k', linestyle='-', linewidth=1.5)
    ax.set_xlim([1, int(Nx/2)])
    ax.set_title('Energy Tendency Budget ($dE/dt$)', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Rate')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize='small', loc='best')
    plt.show()

def plot_febudget_snap(ds, time_idx=-1):
    """Plots Energy Flux Budget from xarray dataset."""
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny)
    
    ds_t = ds.sel(time=time_idx,method='nearest')
    t_val = ds.time.sel(time=time_idx,method='nearest').values
    
    # Load and Normalize terms
    fenl = ds_t.fenlk.values * norm_fac
    feF = -ds_t.fefk.values * norm_fac
    fevisc = -ds_t.fevisck.values * norm_fac
    fefric = -ds_t.fefrick.values * norm_fac
    fefilt = -ds_t.fefiltk.values * norm_fac
    
    # Residual
    fe_sum = fenl + feF + fevisc + fefric + fefilt
    
    ks = ds.k.values / (2 * np.pi / Lx)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.semilogx(ks, fenl, label='NL Flux $\Pi_{NL}$', color='tab:blue', linewidth=2.5)
    ax.semilogx(ks, feF, label='Forcing Flux', color='tab:green', linewidth=2.5)
    ax.semilogx(ks, fevisc, label='Visc. Flux', color='tab:orange', linewidth=2.5)
    ax.semilogx(ks, fefric, label='Fric. Flux', color='tab:red', linewidth=2.5)
    ax.semilogx(ks, fefilt, label='Filt. Flux', color='tab:purple', linewidth=2.5)
    ax.semilogx(ks, fe_sum, 'k--', label='Sum (Residual)', linewidth=1.5)

    ax.axhline(0, color='k', linestyle='-', linewidth=1.5)
    ax.set_xlim([1, int(Nx/2)])
    ax.set_title(f'Energy Flux Budget ($\Pi_E$) (t={t_val:.2f})', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Flux')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize='small', loc='best')
    plt.show()

def plot_febudget_avg(ds, tmin=None,tmax=None,versus=None):
    """Plots Energy Flux Budget from xarray dataset."""
    fscale=16
    Nx, Ny = ds.Nx, ds.Ny
    Lx = ds.x.max().values
    norm_fac = 1.0 / (Nx * Ny)
    
    ds_t = ds.sel(time=slice(tmin,tmax),)
    if versus: ds_t_ref =  versus.sel(time=slice(tmin,tmax),)
    
    # Load and Normalize terms
    fenl = ds_t.fenlk.mean(axis=0).values * norm_fac
    feF = ds_t.fefk.mean(axis=0).values * norm_fac
    fevisc = ds_t.fevisck.mean(axis=0).values * norm_fac
    fefric = ds_t.fefrick.mean(axis=0).values * norm_fac
    fefilt = ds_t.fefiltk.mean(axis=0).values * norm_fac
    if versus:
        fenl_ref = ds_t_ref.fenlk.mean(axis=0).values * norm_fac
        feF_ref = ds_t_ref.fefk.mean(axis=0).values * norm_fac
        fevisc_ref = ds_t_ref.fevisck.mean(axis=0).values * norm_fac
        fefric_ref = ds_t_ref.fefrick.mean(axis=0).values * norm_fac
        fefilt_ref = ds_t_ref.fefiltk.mean(axis=0).values * norm_fac
    
    # Residual
    fe_sum = fenl + feF + fevisc + fefric + fefilt
    if versus:
        fe_sum_ref = fenl_ref + feF_ref + fevisc_ref + fefric_ref + fefilt_ref
    ks = ds.k.values / (2 * np.pi / Lx)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.semilogx(ks, fenl, label='NL Flux $\Pi_{NL}$', color='tab:blue', linewidth=2.5)
    ax.semilogx(ks, feF, label='Forcing Flux', color='tab:green', linewidth=2.5)
    ax.semilogx(ks, fevisc, label='Visc. Flux', color='tab:orange', linewidth=2.5)
    ax.semilogx(ks, fefric, label='Fric. Flux', color='tab:red', linewidth=2.5)
    ax.semilogx(ks, fefilt, label='Filt. Flux', color='tab:purple', linewidth=2.5)
    ax.semilogx(ks, fe_sum, 'k-', label='Sum (Residual)', linewidth=1.5)
    if versus:
        ax.semilogx(ks, fenl_ref,  '--',label='NL Flux $\Pi_{NL} REF$',color='tab:blue', linewidth=2.5)
        ax.semilogx(ks, feF_ref,'--', label='Forcing Flux REF', color='tab:green', linewidth=2.5)
        ax.semilogx(ks, fevisc_ref,'--', label='Visc. Flux REF', color='tab:orange', linewidth=2.5)
        ax.semilogx(ks, fefric_ref,'--', label='Fric. Flux REF', color='tab:red', linewidth=2.5)
        ax.semilogx(ks, fefilt_ref,'--', label='Filt. Flux REF', color='tab:purple', linewidth=2.5)
        ax.semilogx(ks, fe_sum_ref, 'k--', label='Sum (Residual) REF', linewidth=1.5)

    ax.axhline(0, color='k', linestyle='-', linewidth=1.5)
    ax.set_xlim([1, int(Nx/2)])
    ax.set_title('Energy Flux Budget ($\Pi_E$)', fontweight='bold')
    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Flux')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize='small', loc='best')
    plt.show()