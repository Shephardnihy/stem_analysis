import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from matplotlib_scalebar.scalebar import ScaleBar

def PlotDiff(dp, dq, norm = 'log', cmap = cc.cm. fire, vmin_percentile = 1, vmax_percentile = 99.9):
    plt.figure(figsize = (10, 10))
    plt.subplots_adjust(0, 0.15, 1, 1)
    plt.imshow(dp, 
               norm = norm, cmap = cmap, 
               vmin = np.percentile(dp, vmin_percentile), 
               vmax = np.percentile(dp, vmax_percentile))

    ax = plt.gca()
    scalebar = ScaleBar(dq, 
                                 "1/nm",
                                 dimension = "si-length-reciprocal",
                                 location = 'lower right',
                                 frameon = False,
                                 length_fraction = 0.25,    
                                 height_fraction = 0.02,
                                 bbox_to_anchor=(1, -0.12),
                                 bbox_transform =ax.transAxes,
                                 font_properties = {'size': 32},)
    ax.add_artist(scalebar)
    ax.axis("off")
    plt.show()

def PlotImage(image, dr, cmap = 'gray', vmin_percentile = 1, vmax_percentile = 99):
    plt.figure(figsize = (10, 10))
    plt.imshow(image, cmap = cmap, vmin = np.percentile(image, vmin_percentile), vmax = np.percentile(image, vmax_percentile))
    ax = plt.gca()
    ax.axis('off')
    scalebar = ScaleBar(dr, 
                                    "nm",
                                    dimension = "si-length",
                                    location = 'lower right',
                                    frameon = False,
                                    length_fraction = 0.25,    
                                    height_fraction = 0.02,
                                    bbox_to_anchor=(1, -0.12),
                                    bbox_transform =ax.transAxes,
                                    font_properties = {'size': 32},)
    ax.add_artist(scalebar)
    ax.axis("off")
    plt.show()
