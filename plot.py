import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib_scalebar.scalebar import ScaleBar
    import colorcet as cc

    return ScaleBar, np, plt


@app.cell
def _():
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    return


@app.cell
def _():
    import os

    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set Directories
    """)
    return


@app.cell
def _():
    parent_dir = r"D:\FeGaB\2026\For Manuscript"
    return (parent_dir,)


@app.cell
def _(os, parent_dir):
    out_dir = os.path.join(parent_dir, "plots")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir
    return (out_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Data
    """)
    return


@app.cell
def _():
    B_list = [0, 2, 6, 8, 10, 16]
    return (B_list,)


@app.cell
def _(B_list, np, os, parent_dir):
    nn_maps = []
    for B in B_list:
        file_path = os.path.join(parent_dir, f"B{B}", "nn_map.npy")
        nn_map = np.load(file_path)
        nn_maps.append(nn_map)
    nn_maps = np.array(nn_maps)
    return (nn_maps,)


@app.cell
def _(nn_maps):
    nn_maps.shape
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define Helper Function
    """)
    return


@app.cell
def _(ScaleBar, plt):
    def plot_image(img, fig, mpl_kwargs = None, ScaleBar_kwargs= None):
        """
        """
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        if mpl_kwargs is None:
            mpl_kwargs = {}
        
        if ScaleBar_kwargs is None:
            ScaleBar_kwargs = {}
        ax.imshow(img,**mpl_kwargs)
        scalebar = ScaleBar(**ScaleBar_kwargs)
        ax.add_artist(scalebar)
        ax.axis("off")
        return ax

    return (plot_image,)


@app.cell
def _(np, plt):
    def plot_colorbar(cmap, fig, vrange = (0, 1), nlabels = 5, orientation = "vertical", label = None):
        """
        """
        ax = fig.add_subplot(111)
        norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation=orientation)
        if label is not None:
            cbar.set_label(label, fontsize=24)
    
        ticks = np.linspace(vrange[0], vrange[1], nlabels)
        tick_labels = [f"{tick:.2f}" for tick in ticks]
        cbar.ax.set_yticks(ticks, labels=tick_labels, fontsize=24)
        return cbar
    

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot NN maps
    """)
    return


@app.cell
def _():
    color_list = [
        '#d53e4f', 
        '#f46d43', 
        '#66c2a5', 
        '#3288bd', 
        '#abdda4', 
        '#fdae61'
    ]


    return (color_list,)


@app.cell
def _():
    cmap = 'Spectral_r'
    mpl_kwargs = {
        "cmap":cmap,
        "vmin":0.23,
        "vmax":0.26
    }

    scalebar_kwargs = {
        "dx": 2,
        "units": "nm",
        "dimension": "si-length",
        "location": "lower right",
        "length_fraction": 0.25,
        "height_fraction": 0.02,
        "pad": 1,
        "color": "k",
        "font_properties": {"size": 24, "weight": "bold"},
        "frameon": False,
        "scale_loc": "top",
    }
    return cmap, mpl_kwargs, scalebar_kwargs


@app.cell
def _(mpl_kwargs, nn_maps, plot_image, plt, scalebar_kwargs):
    fig = plt.figure(figsize = (10, 10))
    ax = plot_image(nn_maps[2], fig, mpl_kwargs = mpl_kwargs, ScaleBar_kwargs = scalebar_kwargs)
    ax
    return (ax,)


@app.cell
def _(
    B_list,
    mpl_kwargs,
    nn_maps,
    os,
    out_dir,
    plot_image,
    plt,
    scalebar_kwargs,
):
    for ind, Bpercent in enumerate(B_list):
        fig1 = plt.figure(figsize = (10, 10))
        ax1 = plot_image(nn_maps[ind], fig1, mpl_kwargs = mpl_kwargs, ScaleBar_kwargs = scalebar_kwargs)
        plt.savefig(os.path.join(out_dir, f"nn_map_B{Bpercent}.svg"), dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
        plt.close(fig1)
    return


@app.cell
def _(os, out_dir, plt):
    plt.savefig(os.path.join(out_dir, "nn_map_B0.svg"), dpi = 300, bbox_inches = "tight", pad_inches = 0.1)
    return


@app.cell
def _(cmap, np, os, out_dir, plt):
    fig_cbar = plt.figure(figsize = (1, 10))
    ax_cbar = fig_cbar.gca()
    norm = plt.Normalize(vmin=0.23, vmax=0.27)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='vertical',ticklocation='left')

    ticks = np.linspace(0.23, 0.27, 5)
    tick_labels = [f"{tick:.2f}" for tick in ticks]
    cbar.ax.set_yticks(ticks, labels=tick_labels, fontsize=24)
    plt.savefig(os.path.join(out_dir, "nn_map_colorbar.svg"), dpi = 300, bbox_inches = "tight", pad_inches = 0.1)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot Histogram
    """)
    return


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(nn_maps, np):
    nn_maps_flattened = [nn_map.flatten() for nn_map in nn_maps]
    np.shape(nn_maps_flattened)
    return (nn_maps_flattened,)


@app.cell
def _(B_list, ax, color_list, nn_maps_flattened, os, out_dir, plt, sns):
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(left = 0.15, bottom = 0.15, right = 0.85, top = 0.95)
    ax2 = plt.gca()
    sns.histplot(nn_maps_flattened, kde = True, palette = color_list, binrange = (0.23, 0.27), bins = 100, element="step", fill=True, ax = ax2)
    legend = ax.get_legend()
    handles = legend.legend_handles
    ax2.legend(handles, [f'{ratio}% B' for ratio in B_list], fontsize=24)
    ax2.set_xlim(0.23, 0.27)

    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=24)
    #plt.tight_layout()

    ax2.set_xlabel('Nearest Neighbor Distance (nm)', fontsize = 24)
    ax2.set_ylabel('Count', fontsize = 24)
    plt.savefig(os.path.join(out_dir, f"nn_histogram.svg"), dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
    ax2
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Thresholding
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
