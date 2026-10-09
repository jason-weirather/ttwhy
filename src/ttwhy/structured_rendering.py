"""Foreground-only spatial character art over a bounded display raster.

Detail fits eight-dot Braille under a local ink-count constraint. Shape compares
canonical stroke descriptors after a relief-style display transform. No semantic
inference, per-cell background painting, or modification of source pixels occurs.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .glyph_atlas import BRAILLE_BITS, SAMPLE_YX, TONE_ANCHORS, glyph_atlas
from .text_image import TextImage

# Bound matching scratch independently of the terminal size and vocabulary count.
MATCH_BATCH = 256
# Ordered rounding of the fractional dot count, not random pixel noise. This is
# screen-grid anchored; output is reproducible, but not temporally registered.
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                   [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + .5) / 16
# A small, fixed preference arranges nearly equal samples into a balanced pattern.
DOT_TIE = (np.array([0, 4, 6, 2, 1, 5, 7, 3], np.float32) - 3.5) * (.02 / 8)
BRAILLE_CHARACTERS = np.array([' '] + [chr(0x2800 + i) for i in range(1, 256)], dtype='U1')
for _array in (BAYER, DOT_TIE, BRAILLE_CHARACTERS):
    _array.flags.writeable = False


def local_mean(values: NDArray, valid: NDArray, sigma: float) -> NDArray:
    """Normalized convolution: missing coverage is not a fabricated dark neighbor."""
    weight = gaussian_filter(valid.astype(np.float32), sigma, mode='constant')
    total = gaussian_filter(np.where(valid, values, 0), sigma, mode='constant')
    return np.divide(total, weight, out=np.zeros_like(total), where=weight > 1e-7)


def cell_patches(array: NDArray, rows: int, columns: int) -> NDArray:
    """Pack an 8-by-4-per-cell raster into N-by-32 patches, retaining color axes."""
    height, width = SAMPLE_YX
    tail = array.shape[2:]
    return array.reshape(rows, height, columns, width, *tail).transpose(
        0, 2, 1, 3, *range(4, 4 + len(tail))).reshape(rows * columns, height * width, *tail)


def braille_cells(patches: NDArray, coverage: NDArray, rows: int, columns: int) -> NDArray:
    """Fit dot positions while retaining each cell's tone to within one dot.

    A 2-by-2 group of display samples addresses one dot. The desired count is the
    sum of eight dot intensities with ordered fractional rounding. For that count,
    choose the highest intensities (with a small deterministic tie preference).
    This minimizes squared binary reconstruction error for the perturbed samples
    under the count constraint. It is not a perceptual optimum or a learned model.
    """
    values = patches.reshape(-1, 4, 2, 2, 2)
    valid = coverage.reshape(-1, 4, 2, 2, 2)
    count = valid.sum(axis=(2, 4))
    total = values.sum(axis=(2, 4))
    dots = np.divide(total, count, out=np.zeros_like(total), where=count > 0).reshape(-1, 8)
    dot_valid = count.reshape(-1, 8) > 0
    rounding = BAYER[np.arange(rows)[:, None] % 4, np.arange(columns)[None, :] % 4].reshape(-1)
    wanted = np.floor(dots.sum(axis=1) + rounding).astype(np.intp)
    wanted = np.minimum(wanted, dot_valid.sum(axis=1))
    scores = np.where(dot_valid & (dots > 0), dots + DOT_TIE, -np.inf)
    order = np.argsort(-scores, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(8), order.shape), axis=1)
    lit = (rank < wanted[:, None]) & dot_valid
    codes = (lit * BRAILLE_BITS.reshape(8)).sum(axis=1)
    return BRAILLE_CHARACTERS[codes].reshape(rows, columns).copy()


def shape_cells(patches: NDArray, coverage: NDArray, rows: int, columns: int) -> NDArray:
    """Balance tonal error with normalized within-cell stroke correspondence."""
    height, width = SAMPLE_YX
    atlas = glyph_atlas()
    chosen = np.empty(rows * columns, dtype=np.intp)
    for start in range(0, len(patches), MATCH_BATCH):
        stop = min(start + MATCH_BATCH, len(patches))
        batch = patches[start:stop]
        covered = coverage[start:stop]
        count = covered.sum(axis=1, keepdims=True)
        mean = np.divide(batch.sum(axis=1, keepdims=True), count,
                         out=np.zeros((len(batch), 1), np.float32), where=count > 0)
        centered = np.where(covered, batch - mean, 0)
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        contrast2 = np.divide(norm * norm, count, out=np.zeros_like(norm), where=count > 0)
        unit = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-7)
        similarity = np.einsum('ij,kj->ik', unit, atlas.features, optimize=False)
        # Missing positions never influence tone or fabricate shape evidence.
        for row in np.flatnonzero((count[:, 0] > 0) & (count[:, 0] < height * width)):
            masks = atlas.masks.reshape(len(atlas.characters), -1)[:, covered[row]]
            features = masks - masks.mean(axis=1, keepdims=True)
            lengths = np.linalg.norm(features, axis=1)
            dot = features @ unit[row, covered[row]]
            similarity[row] = np.divide(dot, lengths, out=np.zeros_like(dot), where=lengths > 1e-7)
        score = .8 * (mean - atlas.tones) ** 2
        score += 3.6 * contrast2 * (1 - np.clip(similarity, -1, 1))
        # Flat shading should not become unrelated letters. Reserve structural
        # choices for cells with meaningful variation; keep ties deterministic.
        score[contrast2[:, 0] < .0025, TONE_ANCHORS:] = np.inf
        chosen[start:stop] = np.argmin(score, axis=1)
    return atlas.characters[chosen].reshape(rows, columns).copy()


def render_structured(
    pixels: NDArray, *, rows: int, columns: int, mode: str, color: bool, invert: bool,
) -> TextImage:
    """Render validated normalized samples at the renderer's requested sample shape."""
    height, width = SAMPLE_YX
    expected = (rows * height, columns * width)
    if pixels.shape[:2] != expected:
        raise ValueError(f'Structured rendering requires sample shape {expected}')
    if mode not in ('detail', 'shape'):
        raise ValueError('Structured rendering mode must be detail or shape')
    rgb = np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else pixels
    valid = np.isfinite(rgb).all(axis=-1)
    # Peak signal intentionally follows ttwhy's fluorescence composition semantics.
    rgb = np.where(valid[..., None], rgb, 0).astype(np.float32)
    brightness = rgb.max(axis=-1)
    ink = np.where(valid, 1 - brightness if invert else brightness, 0)
    if mode == 'detail':
        smooth = local_mean(ink, valid, 1.2)
        target = np.clip(ink + .65 * (ink - smooth), 0, 1)
    else:
        # Relief drawing: emphasize ridges but retain quieter tonal context.
        # This is an artistic display transform, not a segmentation/edge label.
        broad = local_mean(ink, valid, 3.0)
        target = np.clip(.42 * ink ** 1.35 + 2.2 * np.maximum(ink - broad, 0), 0, 1)
    target = np.where(valid, target, 0)
    patches = cell_patches(target, rows, columns)
    coverage = cell_patches(valid, rows, columns)
    render_cells = braille_cells if mode == 'detail' else shape_cells
    chars = render_cells(patches, coverage, rows, columns)
    chars[~coverage.any(axis=1).reshape(rows, columns)] = ' '

    foreground = None
    if color:
        # Hue belongs to the strokes, never to independent per-cell color tiles.
        total = (cell_patches(rgb, rows, columns) * patches[..., None]).sum(axis=1)
        peak = total.max(axis=-1, keepdims=True)
        hue = np.divide(total, peak, out=np.ones_like(total), where=peak > 1e-7)
        foreground = np.rint(hue * 255).astype(np.uint8).reshape(rows, columns, 3)
    return TextImage(chars, foreground=foreground)
