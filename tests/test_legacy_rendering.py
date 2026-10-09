"""Byte-for-byte goldens captured from the aed55e5 legacy renderer."""

import hashlib

import numpy as np
import pytest

GOLDENS = {
    (False, False, False): "6904e134b8cc21d0f52d9989e3734e2497734f8ff373fd20610ec01d2ff4043f",
    (False, False, True): "6e27fbe7a5afe5094858bf877fd7c3ac48e526154c285cdcca13094e9b2ec300",
    (False, True, False): "6904e134b8cc21d0f52d9989e3734e2497734f8ff373fd20610ec01d2ff4043f",
    (False, True, True): "6e27fbe7a5afe5094858bf877fd7c3ac48e526154c285cdcca13094e9b2ec300",
    (True, False, False): "4a1ae3e216c7411ecc51cba144c550b61a2d6d0496a127a73a9147298834d0f5",
    (True, False, True): "42722309c9854cf848bea27ce27a0f533875e17a2c51494b68d14ba272a17a98",
    (True, True, False): "5c2b8963950c120822358b2c6f146144ecc658fa6674ff3db2f019e27f3b4d1f",
    (True, True, True): "43adac03dbca1ef453f2d215894064bf56c48f81bff22c69fdbd2e994a10c51e",
}


@pytest.mark.parametrize("rgb,color,invert", GOLDENS)
def test_original_ascii_bytes_are_preserved(rgb, color, invert):
    from ttwhy import TextRenderer

    y, x = np.mgrid[:19, :37]
    pixels = ((x * 17 + y * 23) % 256).astype(float) / 255
    if rgb:
        pixels = np.stack((pixels, np.roll(pixels, 3, axis=0),
                           np.roll(pixels, 5, axis=1)), axis=-1)
    pixels[2:4, 5:8] = np.nan
    frame = TextRenderer(mode="simple", color=color, invert=invert).render(
        pixels, rows=11, columns=19)
    digest = hashlib.sha256(frame.to_text(ansi=True).encode()).hexdigest()
    assert digest == GOLDENS[rgb, color, invert]
