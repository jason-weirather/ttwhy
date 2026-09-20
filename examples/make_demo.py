"""Generate deterministic, synthetic tissue-like OME-TIFFs; no patient data.

This is fixture generation, not a biological simulation. tifffile writes the
fixtures independently of the viewer's Omeify read boundary.
"""

from pathlib import Path

import click
import numpy as np
import tifffile


def make_pixels(*, size: int = 768, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Return an RGB brightfield-like field and three marker-like CYX channels."""
    random = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    nx, ny = (x - size / 2) / size, (y - size / 2) / size
    tissue = (nx / .44)**2 + (ny / .38)**2 < 1
    nuclei = np.zeros((size, size), np.float64)
    marker = np.zeros_like(nuclei)
    epithelium = np.zeros_like(nuclei)
    # Gland-like rings provide visible structure at multiple magnifications.
    for cx, cy, radius in [(.30, .33, .11), (.66, .32, .12), (.50, .65, .15)]:
        radius *= size
        distance = np.hypot(x - cx * size, y - cy * size)
        ring = np.exp(-((distance - radius) / (size * .027))**2)
        epithelium = np.maximum(epithelium, ring)
        tissue &= distance > radius * .70
        for angle in np.linspace(0, 2 * np.pi, 90, endpoint=False):
            xx = cx * size + radius * np.cos(angle)
            yy = cy * size + radius * np.sin(angle)
            nuclei = np.maximum(nuclei, np.exp(-((x - xx)**2 + (y - yy)**2) / (size * .007)**2))
    for index in range(430):
        cx, cy = random.uniform(.1, .9, 2) * size
        radius = random.uniform(.0025, .006) * size
        nucleus = np.exp(-((x - cx)**2 + (y - cy)**2) / radius**2)
        nuclei = np.maximum(nuclei, nucleus)
        if index % 3 == 0:
            marker = np.maximum(marker, nucleus)
    nuclei *= tissue
    marker *= tissue
    epithelium *= tissue
    texture = np.clip(.10 + .05 * np.sin(x / 17) * np.cos(y / 21), 0, 1) * tissue
    rgb = np.ones((size, size, 3), np.float64)
    rgb -= (texture + .2 * tissue)[..., None] * np.array([.15, .65, .35])
    rgb -= epithelium[..., None] * np.array([.06, .35, .19])
    rgb -= nuclei[..., None] * np.array([.59, .79, .32])
    fluorescence = np.stack([nuclei, marker, epithelium])
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), np.rint(
        fluorescence * 45000).astype(np.uint16)


def write_demo(path: Path, array: np.ndarray, *, rgb: bool) -> None:
    """Write tiled, deflate-compressed mean pyramids with independent fixture code."""
    metadata = {"axes": "YXS" if rgb else "CYX",
                "Channel": {"Name": ["RGB"] if rgb else ["DAPI", "CD3", "panCK"]},
                "PhysicalSizeX": .5, "PhysicalSizeY": .5,
                "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    options = {"photometric": "rgb" if rgb else "minisblack", "tile": (64, 64),
               "compression": "deflate"}
    levels = [array]
    for _ in range(4):
        last = levels[-1].astype(np.float64)
        if rgb:
            down = (last[::2, ::2] + last[1::2, ::2] + last[::2, 1::2] + last[1::2, 1::2]) / 4
        else:
            down = (last[:, ::2, ::2] + last[:, 1::2, ::2]
                    + last[:, ::2, 1::2] + last[:, 1::2, 1::2]) / 4
        levels.append(np.rint(down).astype(array.dtype))
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(levels[0], subifds=len(levels) - 1, metadata=metadata,
                     resolution=(20000, 20000), resolutionunit="CENTIMETER", **options)
        for index, level in enumerate(levels[1:], 1):
            density = 20000 / 2**index
            writer.write(level, subfiletype=1, resolution=(density, density),
                         resolutionunit="CENTIMETER", **options)


@click.command(context_settings={"show_default": True})
@click.option("--output-directory", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--no-overwrite", is_flag=True)
def main(output_directory: Path, no_overwrite: bool) -> None:
    """Write synthetic-rgb.ome.tif and synthetic-mif.ome.tif into an explicit directory."""
    paths = [output_directory / "synthetic-rgb.ome.tif", output_directory / "synthetic-mif.ome.tif"]
    if no_overwrite and any(path.exists() for path in paths):
        raise click.ClickException("A demo destination already exists")
    output_directory.mkdir(parents=True, exist_ok=True)
    rgb, mif = make_pixels()
    for path, pixels, is_rgb in zip(paths, [rgb, mif], [True, False], strict=True):
        write_demo(path, pixels, rgb=is_rgb)
        click.echo(path)


if __name__ == "__main__":
    main()
