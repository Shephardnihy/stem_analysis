"""
Post-processing helpers that run outside LiberTEM's UDF executor, operating
on the full per-position descriptor array produced by a UDF (analogous to
_gpu.py's role for dpc.py/utils.py).
"""
from collections import deque

import numpy as np

from .UDFs import ComputeDiffractionDescriptor


def _grid_neighbors(y, x, sy, sx, connectivity):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < sy and 0 <= nx < sx:
            yield ny, nx


def cluster_diffraction_patterns(ctx, dataset, cy, cx, rin, rout, threshold,
                                  bin_size=4, log_compress=True,
                                  connectivity=4, min_region_size=1,
                                  roi=None, seed=None, backends=None,
                                  progress=False):
    """
    Cluster the scan positions of a 4D-STEM dataset into spatially contiguous
    groups of similar diffraction-pattern orientation/phase - i.e. segment
    distinct nanocrystalline particles/grains - using seeded region growing
    over cosine similarity between per-position descriptors.

    Each scan position is first reduced to a compact, L2-normalized
    descriptor with ComputeDiffractionDescriptor (direct beam masked out,
    optionally log-compressed, block-averaged). Clusters are then grown by
    picking a random unlabeled position as a seed and repeatedly merging
    spatially-adjacent (grid-neighbor) positions whose descriptor has cosine
    similarity >= threshold to the running mean (centroid) descriptor of the
    region grown so far - comparing to the region's centroid rather than
    only the immediately-preceding neighbor avoids the region slowly
    "chaining" across a gradual boundary into a different grain, one small
    hop at a time. When a region stops growing, a new random seed is picked
    from the remaining unlabeled positions, and this repeats until every
    position has been labeled.

    Cosine similarity is used because it is invariant to per-position
    intensity scale (sample thickness, dose, detector gain drift) and
    instead compares where Bragg peaks fall in the frame - the signal that
    should separate differently-oriented or differently-phased grains.

    Unlike a global all-pairs similarity/clustering approach (O(n^2) time and
    memory in the number of scan positions n), restricting comparisons to
    spatial grid neighbors makes this O(n) in both time and memory: every
    position is labeled (and thus examined) exactly once, and each
    examination only looks at its 4 or 8 grid neighbors, never the rest of
    the dataset. This also guarantees every returned cluster is spatially
    contiguous by construction, matching "one label per particle" - a global
    pairwise approach could otherwise merge two physically separate
    particles that happen to share the same orientation.

    Because clusters are grown from random seeds, results are order/seed
    dependent at ambiguous boundaries (positions roughly equidistant in
    similarity from two growing regions are claimed by whichever reaches
    them first); pass `seed` for reproducibility.

    parameters
    ----------
    ctx : libertem.api.Context

    dataset : libertem.io.dataset.base.DataSet
        A 4D-STEM dataset (2D scan, 2D frame dimensions).

    cy, cx, rin, rout, bin_size, log_compress
        Passed through to ComputeDiffractionDescriptor, see there.

    threshold : float
        Minimum cosine similarity required to merge a spatially-adjacent
        candidate position into a growing region. Higher values give more,
        smaller clusters; lower values give fewer, larger clusters. There is
        no default - pick a value between the typical within-grain and
        across-grain similarity for the dataset at hand.

    connectivity : int
        4 or 8 - which grid neighbors count as spatially adjacent.

    min_region_size : int
        Regions smaller than this (in number of positions) are relabeled to
        -1 instead of kept as a small/noise cluster. Default 1 (no cleanup).

    roi : numpy.ndarray of bool, optional
        Boolean mask over the dataset's navigation shape, as accepted by
        ctx.run_udf. Positions outside the roi are excluded entirely -
        never seeded, never a valid neighbor to merge into - and are
        assigned label -1.

    seed : int or numpy.random.Generator, optional
        Seed (or generator) for the random seed-selection order, for
        reproducible results.

    backends, progress
        Forwarded to ctx.run_udf.

    returns
    -------
    result : dict
        'labels' : (*nav_shape,) int64 array
            Cluster label per scan position, -1 outside roi or below
            min_region_size.
        'descriptors' : (*nav_shape, n_features) float32 array
            The per-position descriptors used for clustering (NaN outside
            roi).
        'n_clusters' : int
            Number of distinct clusters retained (excluding -1).
        'descriptor_buffer' : the raw UDF result object, for callers who
            want LiberTEM's own buffer wrapper (e.g. its .raw_data view).
    """
    udf = ComputeDiffractionDescriptor(
        cy=cy, cx=cx, rin=rin, rout=rout,
        bin_size=bin_size, log_compress=log_compress, normalize=True,
    )
    udf_result = ctx.run_udf(dataset=dataset, udf=udf, roi=roi,
                              backends=backends, progress=progress)
    descriptor_buffer = udf_result['descriptor']
    descriptors = np.asarray(descriptor_buffer.data)
    sy, sx = descriptors.shape[0], descriptors.shape[1]

    valid_mask = ~np.isnan(descriptors).any(axis=-1)
    valid_positions = [(y, x) for y, x in zip(*np.nonzero(valid_mask))]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(valid_positions))
    shuffled = [valid_positions[i] for i in order]

    labels = np.full((sy, sx), -1, dtype=np.int64)
    remaining = set(valid_positions)
    region_sizes = []
    next_label = 0
    cursor = 0

    while remaining:
        while shuffled[cursor] not in remaining:
            cursor += 1
        seed_pos = shuffled[cursor]

        label = next_label
        next_label += 1
        labels[seed_pos] = label
        remaining.remove(seed_pos)
        queue = deque([seed_pos])
        region_sum = descriptors[seed_pos].copy()
        region_count = 1

        while queue:
            pos = queue.popleft()
            centroid_norm = np.linalg.norm(region_sum)
            centroid = region_sum / max(centroid_norm, 1e-12)
            for nbr in _grid_neighbors(pos[0], pos[1], sy, sx, connectivity):
                if nbr in remaining:
                    sim = float(np.dot(centroid, descriptors[nbr]))
                    if sim >= threshold:
                        labels[nbr] = label
                        remaining.remove(nbr)
                        queue.append(nbr)
                        region_sum = region_sum + descriptors[nbr]
                        region_count += 1

        region_sizes.append(region_count)

    if min_region_size > 1:
        for label, size in enumerate(region_sizes):
            if size < min_region_size:
                labels[labels == label] = -1

    n_clusters = len(set(labels[labels >= 0].tolist()))

    return {
        'labels': labels,
        'descriptors': descriptors,
        'n_clusters': n_clusters,
        'descriptor_buffer': descriptor_buffer,
    }
