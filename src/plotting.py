import holoviews as hv
import numpy as np
import jax.numpy as jnp

def plot_light_curve(lc_object, period=None, width=600):
    mjds, mags, errs, fids = lc_object
    plots = []
    for fid, fname in zip(np.unique(fids), ['g', 'r']):
        mask = fids == fid
        x_axis, x_label = mjds[mask], 'Time [MJD]'
        if period is not None:
            x_axis, x_label = np.mod(x_axis, period)/period, 'Phase'
        plots.append(hv.Scatter((x_axis, mags[mask]), kdims=x_label, vdims='Magnitude', label=fname).opts(color=fname))
        plots.append(hv.ErrorBars((x_axis, mags[mask], errs[mask]), 
                                  kdims=x_label, vdims=['Magnitude', 'Error'], label=fname).opts(line_color=fname, color=fname))
    return hv.Overlay(plots).opts(hv.opts.ErrorBars(width=width, lower_head=None, upper_head=None, line_width=2, invert_yaxis=True))
 
def plot_smoothed(pha_interp, mag_interp, err_interp, width=600):
    interp_g_mean = hv.Curve((pha_interp, mag_interp[0]), label='Smooth g').opts(color='g', alpha=0.5)
    interp_r_mean = hv.Curve((pha_interp, mag_interp[1]), label='Smooth r').opts(color='r', alpha=0.5)
    interp_g_err = hv.Spread((pha_interp, mag_interp[0], err_interp[0])).opts(color='g', alpha=0.25)
    interp_r_err = hv.Spread((pha_interp, mag_interp[1], err_interp[1])).opts(color='r', alpha=0.25)
    return hv.Overlay([interp_g_mean, interp_r_mean, interp_g_err, interp_r_err]).opts(hv.opts.Curve(width=width, invert_yaxis=True))

def plot_reconstruction(x_loc0, x_loc1, pha_interp, mag_interp, err_interp, label, width=300):
    predicted_mag_g = hv.Curve((pha_interp, x_loc0), 'Phase', 'Magnitude').opts(color='g', line_width=5) 
    predicted_mag_r = hv.Curve((pha_interp, x_loc1)).opts(color='r', line_width=5) 
    interp_mag_g = hv.Spread((pha_interp, mag_interp[0], err_interp[0])).opts(color='g', alpha=0.25)
    interp_mag_r = hv.Spread((pha_interp, mag_interp[1], err_interp[1])).opts(color='r', alpha=0.25)
    return hv.Overlay([predicted_mag_g, predicted_mag_r, interp_mag_g, interp_mag_r], label=label).opts(hv.opts.Curve(width=width, invert_yaxis=True))

from bokeh.palettes import Category10

def plot_latent_space(z_loc, z_scale, labels_int, le):
    hv.opts.defaults(hv.opts.ErrorBars(lower_head=None, upper_head=None, show_legend=True))

    def plot_class(label, name, color):
        mask = labels_int == label
        center = hv.Scatter((z_loc[mask, 0], z_loc[mask, 1]), 'z1', 'z2', label=name).opts(color=color)
        error_x = hv.ErrorBars((z_loc[mask, 0], z_loc[mask, 1], z_scale[mask, 0]), 
                               horizontal=True).opts(color=color)
        error_y = hv.ErrorBars((z_loc[mask, 0], z_loc[mask, 1], z_scale[mask, 1]), 
                               horizontal=False).opts(color=color)
        return hv.Overlay([center, error_x, error_y])

    variable_star = [plot_class(label, name, color) for label, name, color in zip(np.unique(labels_int), 
                                                                                  le.inverse_transform(np.unique(labels_int)),
                                                                                  Category10[10])]
    return hv.Overlay(variable_star).opts(width=550, height=400, legend_position='right', xlim=(-2, 2), ylim=(-2, 2))

def plot_latent_generation(z1, z2, pha_interp, generate_lc):
    def plot_generation(zi, zj):
        x_loc0, x_loc1 = generate_lc(jnp.stack([zi, zj]))
        g_plot = hv.Curve((pha_interp, x_loc0)).opts(color='g')
        r_plot = hv.Curve((pha_interp, x_loc1)).opts(color='r')
        return hv.Overlay([g_plot, r_plot]).opts(hv.opts.Curve(invert_yaxis=True, xaxis=None, yaxis=None))
    
    recon_2D = {(zi.item(),zj.item()): plot_generation(zi,zj) for zj in z2 for zi in z1}
    return hv.GridSpace(recon_2D, kdims=['z1', 'z2']).opts(plot_size=(60, 60))