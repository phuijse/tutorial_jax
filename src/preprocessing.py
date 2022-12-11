import pickle
import bz2
import numpy as np

def load_ztf_data(path = "../raw_data/lcdata.pbz2"):
    with bz2.BZ2File(path, 'r') as f:
        lcs, periods, labels = pickle.load(f)
    return lcs, periods, labels


def fold(time, period):
    """
    returns phase = time/period - floor(time/period)
    """
    return np.mod(time, period)/period

def kernel_smoothing(lc_data, period, pha_interp, sp=0.15, align=True, normalize=True): 
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
    
    if normalize or align:
        for k, fid in enumerate([1, 2]):
            #idx_max =  np.argmin(mag_interp[k])
            #max_val = np.amax(mag_interp[k] + err_interp[k])
            #min_val = np.amin(mag_interp[k] - err_interp[k])
            if align:
                mag_interp[k] = np.roll(mag_interp[k], -idx_max)
                err_interp[k] = np.roll(err_interp[k], -idx_max)
            if normalize:
                mag_interp[k] = 2*(mag_interp[k] - min_val)/(max_val - min_val) - 1
                err_interp[k] = 2*err_interp[k]/(max_val - min_val)

    return mag_interp, err_interp, [max_val, min_val, idx_max]
