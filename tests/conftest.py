"""Tiny real TIFF fixtures, written independently of Omeify's writer."""

from pathlib import Path

import numpy as np
import pytest
import tifffile


def write_pyramid(path: Path, data: np.ndarray, *, rgb: bool = False, levels: int = 3,
                  compression: str | None = "deflate", tiled: bool = True,
                  byteorder: str = "<", pixel_size: tuple[float, float] | None = (0.5, 0.5)):
    axes = "YXS" if rgb else "YX" if data.ndim == 2 else "CYX"
    count = 1 if rgb or data.ndim == 2 else data.shape[0]
    names = ["RGB"] if rgb else ["DAPI", "CD3", "panCK", "CD68"][:count]
    metadata = {"axes": axes, "Channel": {"Name": names}}
    if pixel_size is not None:
        metadata.update(PhysicalSizeX=pixel_size[0], PhysicalSizeY=pixel_size[1],
                        PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm")
    options = {"photometric": "rgb" if rgb else "minisblack", "compression": compression}
    options.update({"tile": (16, 16)} if tiled else {"rowsperstrip": 7})
    arrays = [data]
    for _ in range(levels - 1):
        # Stored pyramid samples are deliberately independent of ttwhy's resampler.
        last = arrays[-1]
        arrays.append(last[::2, ::2, :] if rgb else last[..., ::2, ::2])
    with tifffile.TiffWriter(path, ome=True, byteorder=byteorder) as writer:
        writer.write(arrays[0], metadata=metadata, subifds=levels - 1, **options)
        for array in arrays[1:]:
            writer.write(array, subfiletype=1, **options)
    return arrays


@pytest.fixture
def image_data():
    y, x = np.mgrid[:65, :97]
    return np.stack([(x + y * 3) * 40, (x > 35) * (y % 13) * 300,
                     ((x - 49)**2 + (y - 32)**2 < 19**2) * 18000]).astype(np.uint16)


@pytest.fixture
def mif(tmp_path, image_data):
    path = tmp_path / "multiplex.ome.tif"
    arrays = write_pyramid(path, image_data)
    return path, arrays


@pytest.fixture
def rgb(tmp_path):
    y, x = np.mgrid[:80, :112]
    image = np.stack([x * 2, y * 3, (x + y) % 256], axis=-1).astype(np.uint8)
    path = tmp_path / "rgb.ome.tif"
    arrays = write_pyramid(path, image, rgb=True)
    return path, arrays
