from libertem.udf import UDF
from skimage.transform import downscale_local_mean, warp_polar
from skimage.feature import canny
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from scipy.ndimage import shift, rotate, center_of_mass
import numpy as np
from .utils import *
from ._gpu import get_ndimage_module
from ._registration import phase_cross_correlation_batch

class PreprocessFRMS6(UDF):
    def __init__(self, gainmap, hardware_bin = 1,software_bin = 1, rotation = 0, shifts = None,  *args, **kwargs):
        """
        Preprocess 4D-STEM dataset, optional binning, rotation and descan correction can be applied.

        Parameters
        ----------
        gainmap : ndarray
            Gain reference of PNCCD. Gainmap will be further binned with harware binning.

        harware_bin : int
            PNCCD acquisition binning, 1 (264*264), 2 (132*132) or 4 (66*66)

        software_bin : int
            Further binning using local mean descale after harware bin

        rotation : float
            Rotation angle between scanning and detector

        shifts : aux_data of ndarray
            shifts needs to be applied to 4D-STEM dataset to compensate for imperfect descanning, should be a (sx, sy, 2) array
        """
        super().__init__(*args, gainmap=gainmap, hardware_bin = hardware_bin, software_bin = software_bin, rotation = rotation, shifts = shifts, **kwargs)

    def get_result_buffers(self):
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        hardware_bin = self.params.hardware_bin
        software_bin = self.params.software_bin

        return {
            "pattern": self.buffer(kind = 'nav', dtype = np.int16, extra_shape = (sx//hardware_bin//software_bin,sy//hardware_bin//software_bin)),
        }

    def process_frame(self, frame):

        hardware_bin = self.params.hardware_bin
        software_bin = self.params.software_bin
        rotation = self.params.rotation
        gainmap = downscale_local_mean(self.params.gainmap, (1, hardware_bin))

        frame_gain_corrected = downscale_local_mean(frame, hardware_bin)*gainmap
        #print(shifts)
        if self.params.shifts is None:
            frame_binned = downscale_local_mean(frame_gain_corrected, software_bin)
            frame_rotated = rotate(frame_binned, rotation, reshape = False)
        else:
            shifts = self.params.shifts[:]
            frame_binned = downscale_local_mean(frame_gain_corrected, software_bin)
            frame_shifted = shift(frame_binned , shifts)
            frame_rotated = rotate(frame_shifted, rotation, reshape = False)


        frame_final = frame_rotated.astype('int16')
        self.results.pattern[:] = frame_final[:]


class FindDiskCenterEllipseFitting(UDF):
    def __init__(self, sigma = 1, low_threshold= 0.3, high_threshold = 1, *args, **kwargs):
        """
        Find 4D-STEM Disk Center using Ellipse Fitting. Ellipse Fitting is useful when diffraction disk has a well-defined edge and only have one disk.
        For example, atomic resolution 4D-STEM dataset on a thin sample or 4D-STEM dataset acquired on vacuum region for descan calibration.

        Parameters
        ----------
        threshold : float
            Intensity threshold to binarized the image for Canny edge detection
        """
        super().__init__(*args, sigma = sigma, low_threshold= low_threshold, high_threshold = high_threshold, **kwargs)

    def get_result_buffers(self):
        return {
            "center": self.buffer(kind = 'nav', dtype = np.float64, extra_shape = (2,)),
        }


    def process_frame(self, frame):
        low_threshold = self.params.low_threshold
        high_threshold = self.params.high_threshold
        sigma = self.params.sigma

        edge = canny(frame, sigma = sigma, low_threshold= low_threshold,high_threshold = high_threshold,use_quantiles=True)
        pts = np.argwhere(edge)
        x, y = pts[:,0],pts[:,1]
        coeffs = FitEllipse(x,y)
        x0, y0, ap, bp, e, phi = Cartesian2Polar(coeffs)
        self.results.center[:] = (x0, y0)


def _choose_batch_frames(working_shape, itemsize=8, budget_bytes=512 * 1024**2, safety_factor=4):
    """
    Pick a per-call frame-batch size for a process_partition loop, given the
    shape of the largest working array computed per frame (e.g. a padded FFT
    buffer), a byte budget, and a safety margin for the several intermediate
    arrays a computation typically holds live at once.
    """
    per_frame = 1
    for s in working_shape:
        per_frame *= s
    per_frame *= itemsize * safety_factor
    return max(1, int(budget_bytes // per_frame))


class VirtualFieldImaging(UDF):
    def __init__(self, cx, cy, rin, rout, *args, **kwargs):
        """
        Calculate virtual field

        parameters
        ----------
        cx : float
            center along x (column)

        cy : float
            center along y (row)

        rin : float
            inner cutoff

        rout : float
            outer cutoff
        """
        super().__init__(*args, cx = cx, cy = cy, rin = rin, rout = rout,**kwargs)

    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {
            "virtual_image": self.buffer(kind = 'nav', dtype = np.float64),
        }

    def get_task_data(self):
        xp = self.xp
        cx, cy, rin, rout = self.params.cx, self.params.cy, self.params.rin, self.params.rout
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        sxx, syy = xp.mgrid[0:sx, 0:sy]
        mask = ((sxx - cx)**2 + (syy - cy)**2 < rout**2) & ((sxx - cx)**2 + (syy - cy)**2 > rin**2)
        batch_frames = _choose_batch_frames((sx, sy))
        return {"mask": mask, "batch_frames": batch_frames}

    def process_partition(self, partition):
        mask = self.task_data.mask
        batch = self.task_data.batch_frames
        n = partition.shape[0]
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = partition[start:end]
            result = (chunk * mask).sum(axis=(1, 2))
            self.results.virtual_image[start:end] = self.forbuf(
                result, self.results.virtual_image[start:end]
            )


class FindDiskShiftCrossCorrelation(UDF):
    def __init__(self, reference_img, upsample_factor = 16, *args, **kwargs):
        """
        Find 4D-STEM Disk Deflection using Cross-Correlation.
        parameters
        ----------
        reference_img : ndarray
            Reference image for cross-correlation
        upsample_factor : int
            subpixel accuracy
        """
        super().__init__(*args, reference_img = reference_img, upsample_factor = upsample_factor)

    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {
            "shift": self.buffer(kind = 'nav', dtype = np.float64, extra_shape = (2,))
        }

    def get_task_data(self):
        xp = self.xp
        reference_img = xp.asarray(self.params.reference_img)
        src_freq = xp.fft.fft2(reference_img)
        batch_frames = _choose_batch_frames(reference_img.shape, itemsize=16, safety_factor=6)
        return {"src_freq": src_freq, "batch_frames": batch_frames}

    def process_partition(self, partition):
        xp = self.xp
        upsample_factor = self.params.upsample_factor
        src_freq = self.task_data.src_freq
        batch = self.task_data.batch_frames
        n = partition.shape[0]
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = partition[start:end]
            s = phase_cross_correlation_batch(xp, src_freq, chunk, upsample_factor=upsample_factor)
            self.results.shift[start:end] = self.forbuf(s, self.results.shift[start:end])


def _sobel_magnitude_batched(xp, ndi, tile):
    """
    Batched port of skimage.filters.sobel(image) (no mask, mode='reflect'):
    gradient magnitude from the two axis-aligned Sobel kernels. `tile` is
    (N, ny, nx); each frame is filtered independently by giving the
    convolution kernel a trivial size-1 leading axis.
    """
    edge = xp.array([1., 0., -1.])
    smooth = xp.array([1., 2., 1.]) / 4
    kernel_0 = xp.outer(edge, smooth)
    kernel_1 = kernel_0.T

    out_0 = ndi.convolve(tile, kernel_0[None, :, :], mode='reflect')
    out_1 = ndi.convolve(tile, kernel_1[None, :, :], mode='reflect')
    return xp.sqrt(out_0**2 + out_1**2) / xp.sqrt(2.0)


def _center_of_mass_batched(xp, tile, coords_y, coords_x):
    """Batched port of scipy.ndimage.center_of_mass for a stack of 2D frames."""
    total = tile.sum(axis=(1, 2))
    cy = (tile * coords_y).sum(axis=(1, 2)) / total
    cx = (tile * coords_x).sum(axis=(1, 2)) / total
    return xp.stack([cy, cx], axis=1)


class FindDiskShiftCOM(UDF):
    def __init__(self, cx, cy, rin, rout = None, ifsobel = False, *args, **kwargs):
        """
        Find 4D-STEM Disk Deflection using Center-of-mass method
        parameters
        ----------
        """
        if rout is None:
            rout = 1000

        super().__init__(*args,cx = cx, cy = cy, rin = rin, rout = rout, ifsobel = ifsobel)

    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {
            "shift": self.buffer(kind = 'nav', dtype = np.float64, extra_shape = (2,))
        }

    def get_task_data(self):
        xp = self.xp
        cx, cy, rin, rout = self.params.cx, self.params.cy, self.params.rin, self.params.rout
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        sxx, syy = xp.mgrid[0:sx, 0:sy]
        mask = ((sxx - cx)**2 + (syy - cy)**2 < rout**2) & ((sxx - cx)**2 + (syy - cy)**2 > rin**2)
        batch_frames = _choose_batch_frames((sx, sy), safety_factor=6)
        return {
            "mask": mask,
            "coords_y": sxx.astype(xp.float64),
            "coords_x": syy.astype(xp.float64),
            "batch_frames": batch_frames,
        }

    def process_partition(self, partition):
        xp = self.xp
        ifsobel = self.params.ifsobel
        mask = self.task_data.mask
        coords_y = self.task_data.coords_y
        coords_x = self.task_data.coords_x
        batch = self.task_data.batch_frames
        n = partition.shape[0]

        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = partition[start:end]

            if ifsobel:
                ndi = get_ndimage_module(xp)
                chunk_filtered = _sobel_magnitude_batched(xp, ndi, chunk)
            else:
                chunk_filtered = chunk

            result = _center_of_mass_batched(xp, chunk_filtered * mask, coords_y, coords_x)
            self.results.shift[start:end] = self.forbuf(result, self.results.shift[start:end])

class GetPolar2D(UDF):
    def __init__(self, radius, center, *args, **kwargs):
        """
        Find 4D-STEM Disk Deflection using Center-of-mass method
        parameters
        ----------
        """
        super().__init__(*args, radius=radius, center=center)

    def get_result_buffers(self):
        return {
            "warped": self.buffer(kind='nav', dtype=np.float32, extra_shape=(360, self.params.radius))
        }

    def process_frame(self, frame):
        warped_frame = warp_polar(frame, center=self.params.center, radius=self.params.radius).astype(np.float32)

        self.results.warped[:] = warped_frame



class CalculateCepstrum(UDF):
    def __init__(self, window = True, pad = True, pad_shape = (1024, 1024), output_shape = (128, 128), *args, **kwargs):
        """
        Cepstrum transform
        parameters
        window : Bool
            If true, a hann window will be applied

        pad : Bool
            If True, the input will be padded

        pad_shape : int
            If pad = True, pad_width specifies the padded frame size

        output_shape : int
            Specify the output shape
        ----------
        """
        super().__init__(*args, window = window, pad = pad, pad_shape = pad_shape, output_shape = output_shape)

    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {
            "cepstrum": self.buffer(kind = 'nav', dtype = np.int16, extra_shape = self.params.output_shape)
        }

    def get_task_data(self):
        xp = self.xp
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]

        w = None
        if self.params.window:
            xx, yy = xp.mgrid[0:sx, 0:sy]
            w = (1/2-1/2*xp.cos(2*np.pi*xx/sx))*(1/2-1/2*xp.cos(2*np.pi*yy/sy))

        pad_width = None
        working_shape = (sx, sy)
        if self.params.pad and self.params.pad_shape:
            py = (self.params.pad_shape[0] - sx)//2
            px = (self.params.pad_shape[1] - sy)//2
            # Matches the original per-frame np.pad(..., (py, px)) call, which
            # applies (before=py, after=px) to both spatial axes.
            pad_width = ((0, 0), (py, px), (py, px))
            working_shape = self.params.pad_shape

        # FFT buffers are complex128 (16 bytes/px); several full-size
        # intermediates (padded, shifted, spectrum, abs) are live at once.
        batch_frames = _choose_batch_frames(working_shape, itemsize=16, safety_factor=6)

        return {"w": w, "pad_width": pad_width, "batch_frames": batch_frames}

    def process_partition(self, partition):
        xp = self.xp
        output_shape = self.params.output_shape
        w = self.task_data.w
        pad_width = self.task_data.pad_width
        batch = self.task_data.batch_frames
        n = partition.shape[0]

        for start in range(0, n, batch):
            end = min(start + batch, n)
            tile = partition[start:end]

            tile = xp.maximum(tile, 1)
            dp_log = xp.log(tile)
            if w is not None:
                dp_log = dp_log * w[None, :, :]

            dp_pad = dp_log if pad_width is None else xp.pad(
                dp_log, pad_width, mode='constant', constant_values=0
            )

            spec = xp.fft.ifftshift(dp_pad, axes=(1, 2))
            spec = xp.fft.fft2(spec, axes=(1, 2))
            spec = xp.fft.fftshift(spec, axes=(1, 2))
            ceps = xp.abs(spec)

            ph, pw = ceps.shape[1], ceps.shape[2]
            y0, y1 = ph//2 - output_shape[0]//2, ph//2 + output_shape[0]//2
            x0, x1 = pw//2 - output_shape[1]//2, pw//2 + output_shape[1]//2
            ceps = ceps[:, y0:y1, x0:x1]

            t = xp.percentile(ceps, 99.99, axis=(1, 2), keepdims=True)
            ceps = xp.minimum(ceps, t)
            mn = ceps.min(axis=(1, 2), keepdims=True)
            mx = ceps.max(axis=(1, 2), keepdims=True)
            ceps = (ceps - mn) / (mx - mn) * 32767

            result = ceps.astype(xp.int16)
            self.results.cepstrum[start:end] = self.forbuf(result, self.results.cepstrum[start:end])


class ComputeDiffractionDescriptor(UDF):
    def __init__(self, cy, cx, rin, rout, bin_size=4,
                 log_compress=True, normalize=True, *args, **kwargs):
        """
        Compute a compact per-position descriptor of the diffraction pattern's
        Bragg-peak layout, for use in downstream clustering of scan positions
        into grains/particles of similar crystal orientation or phase.

        The direct beam is excluded via an annular (ring) mask - like
        VirtualFieldImaging/FindDiskShiftCOM - since it dominates the frame
        intensity and carries no orientation information. The masked region
        is optionally log1p-compressed to tame the dynamic range between the
        direct beam and weaker Bragg reflections, then block-mean-downsampled
        to keep the descriptor length tractable for downstream clustering.

        parameters
        ----------
        cy : float
            Row (first sig axis) coordinate of the diffraction pattern
            center, in pixels.

        cx : float
            Column (second sig axis) coordinate of the diffraction pattern
            center, in pixels.

        rin : float
            Inner radius of the annular mask (pixels); excludes the direct
            beam / central disk.

        rout : float
            Outer radius of the annular mask (pixels).

        bin_size : int
            Block-mean downsampling factor applied to the masked frame
            before flattening to a feature vector.

        log_compress : bool
            If True, apply log1p to the (already masked) intensities before
            downsampling, to reduce the dynamic range of strong vs. weak
            Bragg peaks.

        normalize : bool
            If True, L2-normalize each descriptor to unit norm before
            storing it, so that downstream cosine similarity reduces to a
            plain dot product.
        """
        super().__init__(*args, cy=cy, cx=cx, rin=rin, rout=rout,
                          bin_size=bin_size, log_compress=log_compress,
                          normalize=normalize, **kwargs)

    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        sy, sx = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        bin_size = self.params.bin_size
        n_features = (sy // bin_size) * (sx // bin_size)
        return {
            "descriptor": self.buffer(kind='nav', dtype=np.float32,
                                       extra_shape=(n_features,)),
        }

    def get_task_data(self):
        xp = self.xp
        cy, cx, rin, rout = (self.params.cy, self.params.cx,
                              self.params.rin, self.params.rout)
        sy, sx = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        syy, sxx = xp.mgrid[0:sy, 0:sx]
        r2 = (syy - cy) ** 2 + (sxx - cx) ** 2
        mask = (r2 < rout ** 2) & (r2 > rin ** 2)
        batch_frames = _choose_batch_frames((sy, sx), safety_factor=6)
        return {"mask": mask, "batch_frames": batch_frames}

    def process_partition(self, partition):
        xp = self.xp
        mask = self.task_data.mask
        bin_size = self.params.bin_size
        batch = self.task_data.batch_frames
        n = partition.shape[0]

        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = (partition[start:end] * mask).astype(xp.float32)

            if self.params.log_compress:
                chunk = xp.log1p(xp.maximum(chunk, 0))

            # block-mean downsample: crop any remainder rows/cols that don't
            # divide evenly by bin_size, then reshape+mean over each block.
            ny, nx = chunk.shape[1] // bin_size, chunk.shape[2] // bin_size
            chunk = chunk[:, :ny * bin_size, :nx * bin_size]
            chunk = chunk.reshape(chunk.shape[0], ny, bin_size, nx, bin_size)
            chunk = chunk.mean(axis=(2, 4))

            feat = chunk.reshape(chunk.shape[0], -1)

            if self.params.normalize:
                norm = xp.linalg.norm(feat, axis=1, keepdims=True)
                feat = feat / xp.maximum(norm, 1e-12)

            self.results.descriptor[start:end] = self.forbuf(
                feat, self.results.descriptor[start:end]
            )
