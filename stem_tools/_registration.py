"""
Array-module-generic (numpy or cupy) port of the subset of
skimage.registration.phase_cross_correlation actually used by
stem_tools.UDFs.FindDiskShiftCrossCorrelation, batched over a leading
frame axis so it can process a whole LiberTEM tile in one call.

Only the code path exercised by that UDF is ported: real-space input,
no masks, default "phase" normalization, no disambiguation. See
skimage.registration._phase_cross_correlation.phase_cross_correlation
for the original single-frame algorithm this mirrors.
"""


def _upsampled_dft_batched(xp, data, upsampled_region_size, upsample_factor, axis_offsets):
    """
    Batched matrix-multiply DFT upsampling.

    data : (N, *spatial) complex array
    axis_offsets : (N, ndim) float array, per-frame offsets for each spatial axis
    """
    ndim = data.ndim - 1
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * ndim
    orig_spatial_shape = data.shape[1:]
    im2pi = 1j * 2 * xp.pi

    for axis_idx in reversed(range(ndim)):
        n_items = orig_spatial_shape[axis_idx]
        ups_size = int(upsampled_region_size[axis_idx])
        ax_offset = axis_offsets[:, axis_idx]

        freqs = xp.fft.fftfreq(int(n_items), upsample_factor)
        idx = xp.arange(ups_size)
        kernel = (idx[None, :, None] - ax_offset[:, None, None]) * freqs[None, None, :]
        kernel = xp.exp(-im2pi * kernel).astype(data.dtype)

        # kernel: (N, ups_size, n_items), data: (N, ..., n_items) -> (N, ups_size, ...)
        data = xp.einsum("nij,n...j->ni...", kernel, data)

    return data


def phase_cross_correlation_batch(xp, src_freq, tile, upsample_factor=1):
    """
    Subpixel shift of each frame in `tile` relative to a precomputed reference
    spectrum `src_freq`, batched over the leading (frame) axis.

    xp : numpy or cupy module
    src_freq : (ny, nx) complex array, xp.fft.fft2 of the reference image
    tile : (N, ny, nx) real array
    upsample_factor : int, subpixel accuracy factor

    Returns
    -------
    shift : (N, ndim) float array, one (dy, dx, ...) row per frame
    """
    shape = tile.shape[1:]
    n_frames = tile.shape[0]

    target_freq = xp.fft.fft2(tile, axes=tuple(range(1, tile.ndim)))
    image_product = src_freq[None, ...] * xp.conj(target_freq)
    eps = xp.finfo(image_product.real.dtype).eps
    image_product = image_product / xp.maximum(xp.abs(image_product), 100 * eps)
    cross_correlation = xp.fft.ifft2(image_product, axes=tuple(range(1, tile.ndim)))

    float_dtype = image_product.real.dtype

    abs_cc = xp.abs(cross_correlation).reshape(n_frames, -1)
    maxima_flat = xp.argmax(abs_cc, axis=1)
    maxima = xp.stack(xp.unravel_index(maxima_flat, shape), axis=1).astype(float_dtype)

    midpoint = xp.array([xp.fix(s / 2) for s in shape], dtype=float_dtype)
    shape_arr = xp.array(shape, dtype=float_dtype)
    over = maxima > midpoint[None, :]
    shift = xp.where(over, maxima - shape_arr[None, :], maxima)

    if upsample_factor != 1:
        upsample_factor = float(upsample_factor)
        shift = xp.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = int(xp.ceil(upsample_factor * 1.5))
        dftshift = float(int(upsampled_region_size // 2))
        sample_region_offset = dftshift - shift * upsample_factor

        cc2 = _upsampled_dft_batched(
            xp, xp.conj(image_product), upsampled_region_size, upsample_factor,
            sample_region_offset,
        )
        cc2 = xp.conj(cc2)

        abs_cc2 = xp.abs(cc2).reshape(n_frames, -1)
        maxima2_flat = xp.argmax(abs_cc2, axis=1)
        maxima2 = xp.stack(
            xp.unravel_index(maxima2_flat, (upsampled_region_size,) * len(shape)), axis=1
        ).astype(float_dtype)
        maxima2 = maxima2 - dftshift

        shift = shift + maxima2 / upsample_factor

    for dim in range(len(shape)):
        if shape[dim] == 1:
            shift[:, dim] = 0

    return shift
