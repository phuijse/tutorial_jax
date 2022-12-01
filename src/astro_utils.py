import matplotlib.pyplot as plt
import numpy as np

def plot_params(ax, k, loss, params):
    for ax_ in ax:
        ax_.cla()
    ax[0].plot(range(k), loss[:k])
    for i, name in enumerate(['w', 'b', 's']):
        ax[1].plot(range(k), params[:k, 0, i], label=name)
        ax[2].plot(range(k), params[:k, 1, i], label=name)
    for ax_, name in zip(ax, ['ELBO', 'loc', 'scale (log)']):
        ax_.set_title(name)
    ax[1].legend()
    ax[2].legend()
    
    
def plot_lc(ax, lc):
    mjds, mags, errs, fids = lc
    for fid, c in zip([1, 2], ['g', 'r']):
        mask = fids == fid
        ax.errorbar(mjds[mask], mags[mask], errs[mask], fmt='.', c=c, label=c)
    ax.legend()
    ax.invert_yaxis(); 
    ax.set_xlabel('Modified Julian Date (MJD)\n ')
    ax.set_ylabel('Magnitude\n(The smaller the brighter)');
    
def fold(time, period):
    """
    returns phase = time/period - floor(time/period)
    """
    return np.mod(time, period)/period

def plot_lc_folded(ax, lc, P):
    mjds, mags, errs, fids = lc
    for fid, c in zip([1, 2], ['g', 'r']):
        mask = fids == fid
        phase = fold(mjds[mask], P)
        ax.errorbar(phase, mags[mask], errs[mask], fmt='.', c=c, label=c)
    ax.legend()
    ax.invert_yaxis(); 
    ax.set_xlabel(f'Phase @ Period {P:0.6f}');
    ax.set_ylabel('Magnitude');
    
def plot_lc_features(ax, pha, mag, err):
    for k, c in zip(range(2), ['g', 'r']):
        ax.plot(pha, mag[k], c=c, label=c)
        ax.fill_between(pha, 
                        mag[k] - err[k], 
                        mag[k] + err[k], 
                        color=c, alpha=0.5) 
    ax.legend()
    ax.invert_yaxis();   
    
def featurize_lc(lc_data, period, pha_interp, sp=0.15): 
    mag_interp = np.zeros(shape=(2, len(pha_interp)))
    err_interp = np.zeros(shape=(2, len(pha_interp)))
    mjds, mags, errs, fids = lc_data
                          
    for k, fid in enumerate([1, 2]):
        mask = fids == fid
        pha = fold(mjds[mask], period)
        weight = 1.0/errs[mask]**2
        window = np.exp((np.cos(2.0*np.pi*(pha_interp.reshape(-1,1) - pha)) -1)/sp**2)
        norm = np.sum(weight*window, axis=1) # This can be very small!
        mag_interp[k] = np.sum(weight*window*mags[mask], axis=1)/norm
        dmag = (mag_interp[k] - mags[mask].reshape(-1,1)).T
        err_interp[k] = np.sqrt(np.sum(weight*window*dmag**2, axis=1)/norm)        
        err_interp[k] += np.sqrt(np.median(errs[mask]**2))
        
    idx_max =  np.argmin(mag_interp[0])
    max_val = np.amax(mag_interp[0] + err_interp[0])
    min_val = np.amin(mag_interp[0] - err_interp[0])
    
    for k, fid in enumerate([1, 2]):
        #idx_max =  np.argmin(mag_interp[k])
        #max_val = np.amax(mag_interp[k] + err_interp[k])
        #min_val = np.amin(mag_interp[k] - err_interp[k])
        mag_interp[k] = np.roll(mag_interp[k], -idx_max)
        err_interp[k] = np.roll(err_interp[k], -idx_max)
        mag_interp[k] = 2*(mag_interp[k] - min_val)/(max_val - min_val) - 1
        err_interp[k] = 2*err_interp[k]/(max_val - min_val)
    
    return mag_interp, err_interp, [max_val, min_val, idx_max]

def make_train_plots(ax, nepoch, latent_vars, labels_int, losses):
    for ax_ in ax:
        ax_.cla()
    for k in np.unique(labels_int):
        mask = labels_int == k
        ax[0].errorbar(x=latent_vars[mask, 0], y=latent_vars[mask, 1], 
                       xerr=latent_vars[mask, 2], yerr=latent_vars[mask, 3],
                       fmt='none', alpha=0.5)
    ax[0].set_xlim([-10., 10.])
    ax[0].set_ylim([-10., 10.])
    
    ax[1].plot(range(0, nepoch), losses[:nepoch, 0], label='ELBO train')
    ax[1].plot(range(0, nepoch), losses[:nepoch, 1], label='ELBO valid')
    ax[1].legend()
    ax[1].set_yscale('log')
    