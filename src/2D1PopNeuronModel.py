import numpy as np
import scipy as scipy
from scipy.linalg import circulant
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import signal
from pandas import Series
from random import gauss
import ipywidgets as widgets
from ipywidgets import interact, fixed
import random as rnd
import matplotlib as mpl
from matplotlib.widgets import Slider
from numpy import fft
import scipy.linalg as la          
from scipy import signal
from scipy.stats import norm

def compute_firerate(I, M, h): 
    return np.linalg.solve(I - M, h)

def gaussian(delta, inh_factor):

    return 1./inh_factor**2*np.exp(-delta/2./inh_factor**2)

# analog to 1D Ring with N points > periodic boundary conditions (Torus), kernel is 2D, matrix is block-circulant with circulant blocks (BCCB)
def recurrent_connections_2d(N, a, sigmax, r):

    sigmay = sigmax
    M = N
    
    coord_x, coord_y = np.meshgrid(np.arange(N), np.arange(M))

    deltax = (coord_x[:, :, None, None] - coord_x[None, None, :, :] + N/2)  % N - N/2
    deltay = (coord_y[:, :, None, None] - coord_y[None, None, :, :] + N/2)  % N - N/2

    delta = deltax**2 / sigmax**2 + deltay**2 / sigmay**2

    g1 = gaussian(delta, 1.)/sigmay/sigmax/2./np.pi
    g2 = gaussian(delta, r)/sigmay/sigmax/2./np.pi

    anisotropic_g = (g1 - a*g2)
    kernel = np.real(anisotropic_g)
    kernel = kernel.reshape(N*N, N*N)
    
    return kernel

def compute_influence_2d(kernel, stimulus_locations, stim_strength, N):

    influences = {}
    I = np.eye(N*N)
    
    for stim_loc in stimulus_locations: 
        h = np.zeros(N*N)
        h[stim_loc] = stim_strength
        
        r = compute_firerate(I, kernel, h)
        
        influence = r - h
        influences[stim_loc] = influence
    
    return influences

def compute_full_influence_matrix(kernel, N):
    I = np.eye(N*N)
    R = np.linalg.solve(I - kernel, I)
    influence_matrix = R - I
    
    return influence_matrix

#### VARIANCE ####

def compute_angular_variance_2d(matrix_2d, N, center=None):
    from scipy.ndimage import map_coordinates

    cx, cy = N // 2, N // 2
    max_r = N // 2
    r = np.arange(max_r)

    slices = []
    for angle in range(360):
        angle_rad = np.deg2rad(angle)
        xs = cx + r * np.cos(angle_rad)
        ys = cy + r * np.sin(angle_rad)
        profile = map_coordinates(matrix_2d, [ys, xs], order=1)
        slices.append(profile)

    slices = np.array(slices) 
    variance_per_radius = np.var(slices, axis=0)
    total_variance = np.mean(variance_per_radius)

    X, Y = np.meshgrid(np.arange(N) - cx, np.arange(N) - cy)
    angles_rounded = np.round(np.degrees(np.arctan2(Y, X)) % 360).astype(int) % 360
    radii = np.round(np.sqrt(X**2 + Y**2)).astype(int).clip(0, max_r - 1)

    mean_per_radius = np.mean(slices, axis=0)  
    variance_field = (matrix_2d - mean_per_radius[radii]) ** 2
    variance_slice_0deg = variance_field[cy, cx:] 

    return variance_field, total_variance, slices, variance_slice_0deg

def plot_radial_mean(matrix_2d, N):
    variance_field, total_variance, slices = compute_angular_variance_2d(matrix_2d, N)
    
    mean_per_radius = np.mean(slices, axis=0)  
    r = np.arange(len(mean_per_radius))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, mean_per_radius, color='steelblue', linewidth=2)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Radius $r$', fontsize=12)
    ax.set_ylabel('Mean field value', fontsize=12)
    ax.set_title('Radial Mean — averaged over 360 angles', fontsize=13, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def center_kernel(kernel_2d, col, N):
    row_idx, col_idx = divmod(col, N)
    kernel_2d = np.roll(kernel_2d, N // 2 - row_idx, axis=0)
    kernel_2d = np.roll(kernel_2d, N // 2 - col_idx, axis=1)
    return kernel_2d

def uncenter_field(field_2d, col, N):
    row_idx, col_idx = divmod(col, N)
    field_2d = np.roll(field_2d, row_idx - N // 2, axis=0)
    field_2d = np.roll(field_2d, col_idx - N // 2, axis=1)
    return field_2d

#### PLOT #####

import os

export_dir = '/Users/mschloter/Desktop/crossDominantRegime'
os.makedirs(export_dir, exist_ok=True)

def save_fig(filename):
    plt.savefig(os.path.join(export_dir, filename), dpi=150, bbox_inches='tight')

def plot_kernel(M, columns, N, title, filename=None):
    normC = mpl.colors.TwoSlopeNorm(vcenter=0)
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for i, (ax, column) in enumerate(zip(axes, columns)):
        kernel_2d = np.reshape(M.T[column], (N, N))
        im = ax.imshow(kernel_2d, cmap='bwr', norm=normC)
        
        ax.text(-0.15, 1.05, f'({chr(65+i)})', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top')
        
        ax.set_title(f'{title}\n', fontsize=13)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.6)
        set_imshow_ticks(ax, N)

    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def plot_influence(influence_dict, stimulus_locations, N, title, filename=None):
    normC = mpl.colors.TwoSlopeNorm(vcenter=0)
    n_locs = len(stimulus_locations)
    fig, axes = plt.subplots(1, n_locs, figsize=(6 * n_locs, 5))
    if n_locs == 1:
        axes = [axes]

    for i, (ax, stim_loc) in enumerate(zip(axes, stimulus_locations)):
        influence_2d = np.reshape(influence_dict[stim_loc], (N, N)).T
        stim_y, stim_x = divmod(stim_loc, N)
        im = ax.imshow(influence_2d, cmap='bwr', norm=normC)
        
        ax.text(-0.15, 1.05, f'({chr(65+i)})', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top')
        
        ax.scatter(stim_x, stim_y, color='black', s=80, marker='x', linewidths=2)
        ax.set_title(f'{title}\n', fontsize=13)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.6)
        set_imshow_ticks(ax, N)

    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def plot_angular_variance(variance_field, global_mean, slices, var_slice_0deg, N, title, stim_loc=None, col_loc=None, filename=None, norm=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    variance_field = variance_field.copy()
    var_slice_0deg = var_slice_0deg.copy()
    if np.abs(global_mean) < 1e-9:
        variance_field[:] = 0
        var_slice_0deg[:] = 0
    im = axes[0].imshow(variance_field, cmap='hot_r', norm=norm)
    
    L = N // 2  

    if stim_loc is not None:
        sy, sx = divmod(stim_loc, N)
        axes[0].scatter(sx, sy, color='cyan', s=80,
                    marker='x', linewidths=2, zorder=5)

        x_end = sx + L

        if x_end < N:
            axes[0].plot([sx, x_end], [sy, sy],
                         color='deepskyblue', linestyle='--',
                         linewidth=1.8, alpha=0.9)
        else:
            axes[0].plot([sx, N - 1], [sy, sy],
                     color='deepskyblue', linestyle='--',
                     linewidth=1.8, alpha=0.9)

        
            wrap_end = x_end % N
            axes[0].plot([0, wrap_end], [sy, sy],
                         color='deepskyblue', linestyle='--',
                         linewidth=1.8, alpha=0.9)
      
    if col_loc is not None:
        sy, sx = divmod(col_loc, N)
        x_end = sx + L

        if x_end < N:
            axes[0].plot([sx, x_end], [sy, sy],
                         color='deepskyblue', linestyle='--',
                         linewidth=1.8, alpha=0.9)
        else:
            axes[0].plot([sx, N - 1], [sy, sy],
                     color='deepskyblue', linestyle='--',
                     linewidth=1.8, alpha=0.9)

        
            wrap_end = x_end % N
            axes[0].plot([0, wrap_end], [sy, sy],
                         color='deepskyblue', linestyle='--',
                         linewidth=1.8, alpha=0.9)
   
    
    axes[0].set_title(f'Variance Map\n(Sigma^2 = {global_mean:.4e})', fontsize=15)
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel('y', fontsize=14)
    for spine in axes[0].spines.values():
        spine.set_linewidth(2)
    plt.colorbar(im, ax=axes[0], shrink=0.7)
    set_imshow_ticks(axes[0], N)

    if norm is not None:
        ymin, ymax = norm.vmin, norm.vmax
    else:
        ymin, ymax = variance_field.min(), variance_field.max()
    
    r = np.arange(len(var_slice_0deg))
    axes[1].plot(r, var_slice_0deg, color='steelblue', linewidth=2)
    axes[1].axhline(0, color='gray', linewidth=0.8)
    axes[1].set_ylim(ymin, ymax)  
    axes[1].set_xlabel('Radius (Pixel)', fontsize=12)
    axes[1].set_ylabel('Squared deviation from radial mean', fontsize=12)
    axes[1].set_title('Variance Map — Slice at 0°', fontsize=13)
    axes[1].set_xlim(0, len(var_slice_0deg))
    for spine in axes[1].spines.values():
        spine.set_linewidth(2)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def plot_variance_comparison(var_fields, labels, N, suptitle, stim_loc=None, filename=None, norm=None):
    
    var_fields = [v.copy() for v in var_fields]
    for v in var_fields:
        if np.abs(np.mean(v)) < 1e-9:
            v[:] = 0
    
    if norm is None:
        valid_max = max(v.max() for v in var_fields if np.abs(np.mean(v)) >= 1e-10)
        norm = mpl.colors.Normalize(vmin=0, vmax=valid_max)
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, var_field, label) in enumerate(zip(axes, var_fields, labels)):
        im = ax.imshow(var_field, cmap='hot_r', norm=norm)
        
        ax.text(-0.22, 0.5, f'({chr(65+i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='center')
        
        if stim_loc is not None:
            sy, sx = divmod(stim_loc, N)
            ax.scatter(sx, sy, color='cyan', s=80, marker='x', linewidths=2)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.7)
        set_imshow_ticks(ax, N)
    
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def plot_variance_comparison_diff(var_fields, labels, N, suptitle, stim_loc=None, filename=None, norm=None):
    
    var_fields = [v.copy() for v in var_fields]
    for v in var_fields:
        if np.abs(np.mean(v)) < 1e-9:
            v[:] = 0
    
    if norm is None:
        valid_max = max(v.max() for v in var_fields if np.abs(np.mean(v)) >= 1e-10)
        norm = mpl.colors.Normalize(vmin=0, vmax=valid_max)
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, var_field, label) in enumerate(zip(axes, var_fields, labels)):
        im = ax.imshow(var_field, cmap='RdBu', norm=norm)
        
        ax.text(-0.22, 0.5, f'({chr(65+i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='center')
        
        if stim_loc is not None:
            sy, sx = divmod(stim_loc, N)
            ax.scatter(sx, sy, color='cyan', s=80, marker='x', linewidths=2)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.7)
        set_imshow_ticks(ax, N)
    
    plt.suptitle(suptitle, fontsize=15, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def plot_radius_variance_comparison(var_curves, labels, suptitle, filename=None, shared_ymax=None):
    
    var_curves = [v.copy() for v in var_curves]
    
    for v in var_curves:
        if np.abs(np.mean(v)) < 1e-9:
            v[:] = 0

    global_max = shared_ymax if shared_ymax is not None else max(v.max() for v in var_curves)   
    r = np.arange(len(var_curves[0]))
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for i, (ax, v, label) in enumerate(zip(axes, var_curves, labels)):
        
        ax.plot(r, v, linewidth=2)        
        ax.text(-0.15, 1.05, f'({chr(65+i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Radius r', fontsize=14)
        ax.set_ylabel('Variance', fontsize=14)        
        ax.set_ylim(0, global_max * 1.05)
        ax.set_xlim(0, r[-1])

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.grid(True, alpha=0.3)

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename:
        save_fig(filename)

    plt.show()

def plot_variance_comparison_overlay(var_conn_fields, var_inf_fields, labels, N, suptitle,
                                      stim_loc=None, filename=None, norm_conn=None, norm_inf=None):

    var_conn_fields = [v.copy() for v in var_conn_fields]
    var_inf_fields  = [v.copy() for v in var_inf_fields]

    for v in var_conn_fields + var_inf_fields:
        if np.abs(np.mean(v)) < 1e-9:
            v[:] = 0

    if norm_conn is None:
        valid_max = max((v.max() for v in var_conn_fields if np.abs(np.mean(v)) >= 1e-10), default=1)
        norm_conn = mpl.colors.Normalize(vmin=0, vmax=valid_max)
    if norm_inf is None:
        valid_max = max((v.max() for v in var_inf_fields if np.abs(np.mean(v)) >= 1e-10), default=1)
        norm_inf = mpl.colors.Normalize(vmin=0, vmax=valid_max)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (ax, var_conn, var_inf, label) in enumerate(zip(axes, var_conn_fields, var_inf_fields, labels)):

        im_inf  = ax.imshow(var_inf,  cmap='Blues', norm=norm_inf)
        im_conn = ax.imshow(var_conn, cmap='Reds', norm=norm_conn, alpha = 0.6)

        ax.text(-0.32, 0.5, f'({chr(65 + i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='center')

        if stim_loc is not None:
            sy, sx = divmod(stim_loc, N)
            ax.scatter(sx, sy, color='cyan', s=80, marker='x', linewidths=2)

        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        cb_conn = plt.colorbar(im_conn, ax=ax, shrink=0.4, location='right', pad=0.01)
        cb_conn.set_label('Connectivity', fontsize=9)
        cb_inf  = plt.colorbar(im_inf,  ax=ax, shrink=0.4, location='bottom', pad=0.18)
        cb_inf.set_label('Influence', fontsize=9)

        set_imshow_ticks(ax, N)

    plt.suptitle(suptitle, fontsize=15, fontweight='bold')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.08)
    if filename:
        save_fig(filename)
    plt.show()

def set_first_last_ticksize(ax, fontsize_default=10, fontsize_edge=13):
    plt.draw()
    for get_labels in [ax.get_xticklabels, ax.get_yticklabels]:
        labels = [l for l in get_labels() if l.get_text() != '']
        if len(labels) >= 2:
            for label in labels:
                label.set_fontsize(fontsize_default)
            labels[0].set_fontsize(fontsize_edge)
            labels[-1].set_fontsize(fontsize_edge)


def set_imshow_ticks(ax, N, fontsize_default=10, fontsize_edge=13):
    ticks = [0, N//4, N//2, 3*N//4, N-1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    plt.draw()
    for get_labels in [ax.get_xticklabels, ax.get_yticklabels]:
        labels = get_labels()
        for label in labels:
            label.set_fontsize(fontsize_default)
        labels[0].set_fontsize(fontsize_edge)
        labels[-1].set_fontsize(fontsize_edge)

def plot_radius_variance_all_locations(
    var_curves_per_location,
    pert_labels,
    suptitle,
    filename=None,
    shared_ymax=None
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    subplot_labels = ['(A)', '(B)', '(C)', '(D)']
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for ax, pert_label, subplot_label in zip(axes, pert_labels, subplot_labels):
        for (loc_label, curves), color in zip(var_curves_per_location.items(), colors):
            idx = pert_labels.index(pert_label)
            r = np.arange(len(curves[idx]))
            ax.plot(r, curves[idx], linewidth=2, label=loc_label, color=color)

        ax.set_title(f'{subplot_label}  {pert_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Radius r', fontsize=13)
        ax.set_ylabel('Variance', fontsize=13)
        ax.set_xlim(0, len(r))
        if shared_ymax is not None:
            ax.set_ylim(0, shared_ymax)
        ax.tick_params(axis='y', labelleft=True)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename: save_fig(filename)
    plt.show()

def fftind(size):
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )
    
def gaussian_random_field(alpha = 0,
                          size = N,
                          flag_normalize = True):
    k_idx = fftind(size)
    amplitude = np.power(k_idx[0]**2 + k_idx[1]**2 ,- alpha/2)
    amplitude[0,0] = 0
    noise = np.random.normal(size = (size, size)) \
       + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise*amplitude).real
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)

    return gfield

def Func(N=50, seed=42, alpha_grf=3):
    D = int(N*N/2)
    sigmax = 1.5
    sigmay = sigmax
    coord_x, coord_y = np.meshgrid(np.arange(N), np.arange(N))
    deltax = (coord_x[:,:,None,None] - coord_x[None,None,:,:] + N/2) % N - N/2
    deltay = (coord_y[:,:,None,None] - coord_y[None,None,:,:] + N/2) % N - N/2
    deltax = np.abs(deltax)
    deltay = np.abs(deltay)
    delta  = deltax**2/sigmax**2 + deltay**2/sigmay**2
    Gauss  = gaussian(delta, 1.) / sigmay / sigmax / 2. / np.pi
    Gauss  = Gauss.reshape(N*N, N*N)
    max_ev = np.nanmax(np.real(la.eigvals(Gauss)))
    Gauss  = (0.95 / max_ev) * Gauss

    np.random.seed(seed)
    grf = gaussian_random_field(alpha=alpha_grf, size=N)  

    kernel = np.fft.fftshift(np.reshape(Gauss.T[D], (N, N)))
    grf_fft    = np.fft.fft2(grf)
    kernel_fft = np.fft.fft2(kernel)

    G = np.zeros((N*N, N*N))
    for i in range(N*N):
        kernel_i_fft = kernel_fft * np.exp(
            2j * np.pi * (
                np.fft.fftfreq(N)[:, None] * (i // N) +
                np.fft.fftfreq(N)[None, :] * (i  % N)
            )
        )
        c1 = np.real(np.fft.ifft2(grf_fft * kernel_i_fft))
        G[:, i] = c1.ravel()

    return G, grf

def zero_if_small(v, threshold=1e-9):
        v = v.copy()
        if np.abs(np.mean(v)) < threshold:
            v[:] = 0
        return v

def plot_variance_diff_across_locations(results, col_stim_pairs, loc_names, pert_idx,
                                         pert_label, N, filename=None):

    n_locs = len(loc_names)

    diffs = [results[(col, stim_loc)][pert_idx]['var_inf'] -
             results[(col, stim_loc)][pert_idx]['var_conn']
             for col, stim_loc in col_stim_pairs]
    vmax = max(abs(d).max() for d in diffs)
    norm_diff = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, n_locs, figsize=(6 * n_locs, 5))
    if n_locs == 1:
        axes = [axes]

    for i, (ax, (col, stim_loc), loc_name, diff) in enumerate(
            zip(axes, col_stim_pairs, loc_names, diffs)):

        im = ax.imshow(diff, cmap='RdBu_r', norm=norm_diff)

        ax.text(-0.15, 1.05, f'({chr(65 + i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        sy, sx = divmod(stim_loc, N)
        ax.scatter(sx, sy, color='black', s=80, marker='x', linewidths=2)

        ax.set_title(loc_name, fontsize=13)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.6)
        set_imshow_ticks(ax, N)

    plt.suptitle(f'Differenz Influence–Connectivity – {pert_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename:
        save_fig(filename)
    plt.show()

def plot_variance_comparison_across_locations(results, col_stim_pairs, loc_names, pert_idx,
                                               pert_label, N, norm, key='var_inf',
                                               filename=None):
    n_locs = len(loc_names)
    fig, axes = plt.subplots(1, n_locs, figsize=(6 * n_locs, 5))
    if n_locs == 1:
        axes = [axes]

    for i, (ax, (col, stim_loc), loc_name) in enumerate(zip(axes, col_stim_pairs, loc_names)):
        r = results[(col, stim_loc)][pert_idx]
        var_field = r[key]

        im = ax.imshow(var_field, cmap='hot_r', norm=norm)

        ax.text(-0.15, 1.05, f'({chr(65 + i)})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        if key == 'var_inf':
            sy, sx = divmod(stim_loc, N)
            ax.scatter(sx, sy, color='cyan', s=80, marker='x', linewidths=2)
        elif key == 'var_conn':
            sy, sx = divmod(col, N)
            #ax.scatter(sx, sy, color='cyan', s=80, marker='x', linewidths=2)

        ax.set_title(loc_name, fontsize=13)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, shrink=0.6)
        set_imshow_ticks(ax, N)

    plt.suptitle(f'{key.replace("_", " ").title()} – {pert_label}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(pad=3.0)
    if filename:
        save_fig(filename)
    plt.show()
