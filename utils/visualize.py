import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as mpatches

import seaborn as sns
import PIL.Image as Image

import os
import numpy as np
import torch

from zennit.attribution import Gradient, SmoothGrad
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
from zennit.image import interval_norm_bounds
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
import pandas as pd

def draw_radar_plot(data: pd.DataFrame, theta: str, r: str | list[str], hue=None, errorbar=False,
                    save_path="radar_plot.png", title="", refresh=False, subtitle=None):
    # r is column names

    if os.path.exists(save_path) and (not refresh):
        print("Skipped "+save_path+", file already exists.")
        return 0

    thetas = sorted(set(data[theta]), key=list(data[theta]).index)
    series = sorted(set(data[hue]), key=list(data[hue]).index)

    if (len(thetas) == 0) or (len(series) == 0):
        return 0

    if "average" in thetas:
        data = data[data[theta] != "average"].copy()
        thetas.remove("average")
    
    theta_ = np.linspace(0.0, 2 * np.pi, len(thetas), endpoint=False)
    theta_ = np.append(theta_, theta_[0])
    colors = list(plt.colormaps["tab10"].colors)

    ncols = 3
    if type(r) is not list:
        r = [r]
    nrows = int(np.ceil(len(r) / 3.0))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,8*nrows),
                             subplot_kw={'projection': 'polar'}, layout="constrained")
    
    axes = axes.flatten()
    r_max = 0
    r_min = 0
    for ax_i, ax in enumerate(axes):
        if ax_i >= len(r):
            ax.set_axis_off()
            continue

        r_ = r[ax_i]
        for s_i, s in enumerate(series):
            series_j = data[data[hue]==s].copy()
            radii = series_j[r_+"_mean"].to_numpy()
            radii = np.append(radii, radii[0])
            r_max = max(r_max, radii.max())
            r_min = min(r_min, radii.min())

            ax.plot(theta_, radii, linewidth=1, linestyle='solid', label=s, color=colors[s_i])
            ax.fill(theta_, radii, color=colors[s_i], alpha=0.1)
            
            if (r_+"_std" in series_j.columns) and errorbar:
                y_err = series_j[r_+"_std"].to_numpy()
                y_err = np.append(y_err, y_err[0])
                r_max = max(r_max, (radii+y_err).max())
                r_min = min(r_min, (radii-y_err).min())
                ax.errorbar(theta_, radii, yerr=y_err, capsize=7, fmt="o", markersize=3, color=colors[s_i])
        
        theta__ = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
        theta__ = np.append(theta__, theta__[0])
        ax.plot(theta__, np.zeros_like(theta__), color="k")
        ax.set_xticks(theta_[:-1], thetas)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        if subtitle is None:
            ax.set_title(r_.replace("dae", "DAE rate").replace("mae", "MAE rate").replace("_", " - "))
        else:
            ax.set_title(subtitle[ax_i])

    for ax_i, ax in enumerate(axes):
        ax.set_rlim([r_min,r_max])

    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()

def draw_line_chart(data: pd.DataFrame, x: str, y: str | list[str], hue=None, errorbar=False,
                    save_path="line_chart.png", title="", refresh=False, subtitle=None):

    if os.path.exists(save_path) and (not refresh):
        print("Skipped "+save_path+", file already exists.")
        return 0

    xs = sorted(set(data[x]), key=list(data[x]).index)
    series = sorted(set(data[hue]), key=list(data[hue]).index)

    if (len(xs) == 0) or (len(series) == 0):
        return 0

    if "average" in xs:
        data = data[data[x] != "average"].copy()
        xs.remove("average")
    
    x_ = np.arange(len(xs))
    colors = list(plt.colormaps["tab10"].colors)

    ncols = 2
    if type(y) is not list:
        y = [y]
    nrows = int(np.ceil(len(y) / float(ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 4*nrows),
                             layout="constrained")
    
    axes = axes.flatten()
    y_max = 0
    y_min = 0
    for ax_i, ax in enumerate(axes):
        if ax_i >= len(y):
            ax.set_axis_off()
            continue

        y_ = y[ax_i]
        for s_i, s in enumerate(series):
            series_j = data[data[hue]==s].copy()
            y_mean = series_j[y_+"_mean"].to_numpy()
            y_max = max(y_max, y_mean.max())
            y_min = min(y_min, y_mean.min())

            ax.plot(x_, y_mean, linewidth=1, label=s, color=colors[s_i], marker="o", markersize=5)
            
            if (y_+"_std" in series_j.columns) and errorbar:
                y_err = series_j[y_+"_std"].to_numpy()
                ax.fill_between(x_, y_mean-y_err, y_mean+y_err, color=colors[s_i], alpha=0.1)
                y_max = max(y_max, (y_mean+y_err).max())
                y_min = min(y_min, (y_mean-y_err).min())

        ax.axhline(y=0, xmin=0, xmax=max(x_), color="k", linestyle="--")
        ax.set_xticks(x_, xs, rotation=45)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        if subtitle is None:
            ax.set_title(y_.replace("dae", "DAE rate").replace("mae", "MAE rate").replace("_", " - "))
        else:
            ax.set_title(subtitle[ax_i])

    for ax_i, ax in enumerate(axes):
        ax.set_ylim([y_min,y_max])

    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def save_images(img_arrays, figname, n_max=5):

    fig, axes = plt.subplots(1, n_max, figsize=(n_max*2, 2), facecolor="white")
    permuted_indices = torch.randperm(img_arrays.size(0))
    img_arrays = img_arrays[permuted_indices][:n_max]

    for i, ax in enumerate(axes):
        if i < img_arrays.size(0):
            img  = Image.fromarray(np.uint8(img_arrays[i]*255), mode='RGB')
            ax.imshow(img)
        #ax2.set_title("ID image", size=10, color='b')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(figname)
    plt.close()

def save_image_examples(images_dict, figname, n_row=5):
    assert len(images_dict) >= 2, "At least 2 types of samples are required"

    fig, axes = plt.subplots(len(images_dict), n_row, figsize=(n_row*2, len(images_dict)*2), facecolor="white")

    nums = []
    for v in images_dict.values():
        if type(v) is tuple:
            nums.append(len(v[0]))
        else:
            nums.append(len(v))
    percent = nums / np.sum(nums) * 100

    for i, (k, v) in enumerate(images_dict.items()):
        
        if type(v) is tuple:
            lbs = v[1]
            imgs = v[0]
        else:
            lbs = None
            imgs = v

        num = len(imgs)
        permuted_indices = torch.randperm(num)
        imgs = imgs[permuted_indices][:n_row]

        for j in range(n_row):
            ax = axes[i][j]
            if j < imgs.size(0):
                img  = Image.fromarray(np.uint8(imgs[j]*255), mode='RGB')
                ax.imshow(img)
                if lbs is not None:
                    if j == 0:
                        ax.set_title(k+f" ({round(percent[i],2)}%)"+f"\ny={int(lbs[j])}", size=10, color='b')
                    else:
                        ax.set_title(f"\ny={int(lbs[j])}", size=10, color='b')
                elif j == 0:
                    ax.set_title(k+f" ({round(percent[i],2)}%)", size=10, color='b')

            ax.axis('off')
        
    fig.tight_layout()
    fig.savefig(figname)
    plt.close()

def show_images(x, save_path=None, cmap='viridis', vmax=1.0, vmin=0.0, title=""):

    x = np.array(x)
    if x.ndim == 3:
        x = x[None]
    elif x.ndim != 4:
        return -1
    
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 1.5), facecolor="white", layout="constrained")
    axes = axes.flat
    for i, ax in enumerate(axes):
        if len(x) > i:
            img = np.transpose(x[i], (1,2,0))
            if img.shape[-1] == 1:
                img.squeeze()
            # print(img.shape, img)
            ax.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin)
        ax.axis("off")

    plt.suptitle(title)
    if save_path is not None:
        folder_path = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(save_path)
        
    plt.show()
    plt.close()

# Make LRP heatmaps
def make_lrp_heatmap(model, input_normalizer, x, y_true, device="cpu", show_image=False):
    
    if show_image:
        from matplotlib.colors import LinearSegmentedColormap

        bkr_data = [(0.0, (0.0, 1.0, 1.0)), (0.25, (0.0, 0.0, 1.0)), (0.5, (0.0, 0.0, 0.0)),
                    (0.67, (1.0, 0.0, 0.0)),(0.84, (1.0, 1.0, 0.0)),(1.0, (1.0, 1.0, 1.0))]

        cmap = LinearSegmentedColormap.from_list("bkr", bkr_data)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    
    # use the ResNet-specific canonizer
    canonizer = ResNetCanonizer()

    # the ZBox rule needs the lowest and highest values, which are here for
    # ImageNet 0. and 1. with a different normalization for each channel
    low, high = input_normalizer(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

    # create a composite, specifying the canonizers, if any
    composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])

    # choose a target class for the attribution (label 437 is lighthouse)
    target = torch.eye(100)[[y_true]].to(device)

    # create the attributor, specifying model and composite
    with Gradient(model=model, composite=composite) as attributor:
        # compute the model output and attribution
        data = input_normalizer(x.to(device))
        output, attribution = attributor(data, target)
        
    torch.cuda.empty_cache()
    
    attribution = attribution.cpu()
    # sum over the channels
    relevance = np.array(attribution.sum(1))
    relevance = relevance[:, None]

    dims = tuple(range(relevance.ndim))
    vmin, vmax = interval_norm_bounds(relevance, symmetric=True, dim=dims)

    # normalize
    relevance = (relevance - vmin) / (vmax - vmin)

    if show_image:
        show_images(relevance, cmap=cmap, vmax=1.0, vmin=0.0)
    
    return relevance


def interpol_contour(x, y, z, xlim=None, ylim=None, title="", s=1, true_labels=None, marker_mask=None, 
                     thresholds=[], label_class_dict=None, marker_panel=None, save_path="contour_map.png"):
    """
    This function visualizes the OoD scores distribution and the decision boundary of OoD detectors using
    contour maps.
    """
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    color_panel = {0: 'gray', 1: 'red', 2: 'green', 3: 'yellow', 4: 'orange', 
                   5: 'purple', 6: 'pink', 7: 'brown', 8: 'cyan', 9: 'magenta',
                   -2: 'black', -1: 'blue'}
    p_colors = []
    markers = []
    if true_labels is not None:
        p_colors = [color_panel[int(lb)] for lb in true_labels]
    if marker_mask is not None:
        markers = [marker_panel[msk] for msk in marker_mask]
    
    # -----------------------
    # Interpolation on a grid

    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.
    # -----------------------

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    # Create grid values.
    ngridx = 1000
    ngridy = 1000
    npts = len(z)
    xi = np.linspace(min(x), max(x), ngridx)
    yi = np.linspace(min(y), max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    cntr1 = plt.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
    plt.contour(xi, yi, zi, levels=14, linewidths=0.2, colors='k')
    # Show threshold using contour lines
    if len(thresholds):
        contour = plt.contour(xi, yi, zi, levels=thresholds, colors='k', linestyles='dashed', linewidths=2)
        # if (xlim is not None) and (ylim is not None):
        # plt.clabel(contour, inline=1, fontsize=10, colors='k')

    fig.colorbar(cntr1)

    if len(p_colors):
        if len(markers):
            for mk in set(markers):
                idx = np.where(np.array(markers)==mk)[0]
                plt.scatter(x[idx], y[idx], c=np.array(p_colors)[idx], s=s, marker=mk)
        else:
            plt.scatter(x, y, c=np.array(p_colors), s=s)
            
    # Create a legend for color and marker based on color_panel and marker_panel
    handles = []      
    if label_class_dict is not None:
        color_patches = [mpatches.Patch(color=color_panel[key], label=f'{label_class_dict[key]}') 
                         for key in color_panel if key in label_class_dict]
        handles += color_patches
    if marker_panel is not None:
        marker_handles = [plt.Line2D([], [], marker=marker_panel[key], color='k', 
                                     markersize=10, label=key) for key in marker_panel]
        handles += marker_handles
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.3, 1))
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()
    if not os.path.exists("./images/"):
        os.makedirs("./images/")
    plt.savefig(save_path)
    plt.close("all")