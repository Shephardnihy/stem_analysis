import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.widgets import Button, Slider, RangeSlider
from skimage.filters import window

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


class PtychoSliceViewer:
    def __init__(
            self,
            img_stack
    ):
        self.img_stack = img_stack
        self.num_slices = img_stack.shape[0]
        self.image_shape = img_stack.shape[1:]
        w_2d = window('hann', self.image_shape)
        self.fft_slice_by_slice_abs = np.array([np.abs(np.fft.fftshift(np.fft.fft2(img_stack[i]*w_2d))) for i in range(self.num_slices)])
        self.img_vmin, self.img_vmax = img_stack.min(), img_stack.max()
        self.fft_vmin, self.fft_vmax = self.fft_slice_by_slice_abs.min(), self.fft_slice_by_slice_abs.max()

        self.fig, (self.ax_img, self.ax_fft) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.subplots_adjust(bottom = 0.25, left = 0.13, right = 0.78)

        for ax in (self.ax_img, self.ax_fft):
            ax.set_axis_off()

        frame_ax = self.fig.add_axes((0.18, 0.16, 0.64, 0.03))
        left_range_ax = self.fig.add_axes((0.1, 0.25, 0.0225, 0.63))
        right_range_ax = self.fig.add_axes((0.8, 0.25, 0.0225, 0.63))
        reset_ax = self.fig.add_axes((0.84, 0.13, 0.10, 0.05))

        self.img = self.ax_img.imshow(self.img_stack[0], vmin=self.img_vmin, vmax=self.img_vmax)
        self.fft_img = self.ax_fft.imshow(self.fft_slice_by_slice_abs[0], vmin=self.fft_vmin, vmax=self.fft_vmax, norm = 'log')

        self.frame_slider = Slider(
            frame_ax,
            "Frame",
            valmin=0,
            valmax=self.num_slices-1,
            valinit=0,
            valstep=1,
        )

        self.img_vlim = RangeSlider(
            left_range_ax,
            "vlim",
            valmin=float(self.img_vmin),
            valmax=float(self.img_vmax),
            valinit=(self.img_vmin, self.img_vmax),
            orientation = 'vertical'
        )

        self.fft_vlim = RangeSlider(
            right_range_ax,
            "vlim",
            valmin=float(self.fft_vmin),
            valmax=float(self.fft_vmax),
            valinit=(self.fft_vmin, self.fft_vmax),
            orientation = 'vertical'
        )

        self.reset_button = Button(
            reset_ax,
            "Reset",
        )

        self.frame_slider.on_changed(self._update)
        self.img_vlim.on_changed(self._update)
        self.fft_vlim.on_changed(self._update)
        self.reset_button.on_clicked(self._reset)

        self._update()

    def _update(self, _=None):
        frame = int(self.frame_slider.val)
        self.img.set_data(self.img_stack[frame])
        self.fft_img.set_data(self.fft_slice_by_slice_abs[frame])

        self.img.set_clim(*self.img_vlim.val)
        self.fft_img.set_clim(*self.fft_vlim.val)

        self.fig.suptitle(f"Frame {frame}")
        self.fig.canvas.draw_idle()

    def _reset(self, _event):
        self.frame_slider.reset()
        self.img_vlim.reset()
        self.fft_vlim.reset()

    def show(self):
        plt.show()