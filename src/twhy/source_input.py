"""One bounded remote-file staging boundary; Zarr stays regional and untouched.

A downloaded single-file source must outlive interactive PDF subprocesses,
page navigation and native text extraction. The context manager owns precisely
that lifetime. No downloader or filesystem client is imported by the array renderer.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

DEFAULT_MAX_REMOTE_MIB = 128.0
_BLOCK = 1024 * 1024
_REMOTE_SCHEMES = {"http", "https", "s3"}


def remote_scheme(value: str | Path) -> str | None:
    text = str(value)
    if "://" not in text:
        return None
    scheme = urlsplit(text).scheme.lower()
    if scheme not in _REMOTE_SCHEMES:
        raise ValueError("Remote input supports HTTP, HTTPS and s3:// URLs only")
    return scheme


def validate_url(value: str | Path, *, zarr: bool = False) -> str:
    """Require a store/file URL, never accept embedded user:password syntax."""
    text = str(value)
    scheme = remote_scheme(text)
    if scheme is None:
        raise ValueError("Expected an HTTP(S) or s3:// URL")
    url = urlsplit(text)
    if not url.hostname or url.username is not None or url.password is not None or url.fragment:
        raise ValueError(
            "A remote URL needs a host/bucket and must not contain credentials or fragments"
        )
    if scheme == "s3" and (url.port is not None or url.query or
                          any(part in {".", ".."} for part in unquote(url.path).split("/"))):
        raise ValueError("Use an s3://bucket/key URI without queries or traversal segments")
    if zarr and url.query:
        raise ValueError(
            "Zarr store roots cannot have query parameters; configure credentials separately"
        )
    if scheme == "s3" and not zarr and not url.path.strip("/"):
        raise ValueError("An S3 input must identify a store or object key under its bucket")
    return text.rstrip("/") if zarr else text


def display_name(value: str | Path) -> str:
    """Return only the harmless basename, never a signed URL or bucket credentials."""
    scheme = remote_scheme(value)
    if scheme:
        name = unquote(urlsplit(str(value)).path.rstrip("/").rsplit("/", 1)[-1])
    else:
        name = Path(value).name
    name = "".join(char if char.isprintable() and char not in "\\/" else "_"
                   for char in name)
    return (name[:120] or "Remote image")


def is_zarr_input(value: str | Path, *, input_format: str = "auto") -> bool:
    if input_format not in {"auto", "zarr"}:
        raise ValueError("input_format must be auto or zarr")
    if input_format == "zarr":
        return True
    if remote_scheme(value):
        return urlsplit(str(value)).path.rstrip("/").casefold().endswith(".zarr")
    path = Path(value)
    return path.is_dir() and (path.name.casefold().endswith(".zarr") or
                              (path / "zarr.json").is_file() or (path / ".zgroup").is_file())


def s3_options(endpoint_url: str | None = None, *, anonymous: bool = False) -> dict:
    """Use the standard AWS credential chain, optionally targeting local S3-compatible storage."""
    value = endpoint_url if endpoint_url is not None else os.environ.get("AWS_ENDPOINT_URL_S3")
    if not value:
        return {"anon": True} if anonymous else {}
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("S3 endpoint must be a plain HTTP(S) origin without credentials or a path")
    options = {"client_kwargs": {"endpoint_url": value.rstrip("/")}}
    if anonymous:
        options["anon"] = True
    return options


@dataclass(frozen=True)
class ResolvedInput:
    path: Path | str
    label: str
    is_zarr: bool
    remote: bool


def _copy_bounded(source, destination: Path, maximum: int) -> None:
    written = 0
    with destination.open("xb") as out:
        while True:
            block = source.read(min(_BLOCK, maximum - written + 1))
            if not block:
                break
            written += len(block)
            if written > maximum:
                raise ValueError(
                    "Remote file exceeds --max-remote-mib; increase the limit explicitly"
                )
            out.write(block)
    if not written:
        raise ValueError("Remote file is empty")


def _download(url: str, destination: Path, maximum: int, *, endpoint_url: str | None,
              s3_anonymous: bool = False) -> None:
    scheme = remote_scheme(url)
    if scheme == "s3":
        try:
            import s3fs  # noqa: F401, the optional S3 transport
            import fsspec
        except ImportError as error:
            raise RuntimeError('S3 input requires: python -m pip install "twhy[s3]"') from error
        # One object stream, not a whole-bucket listing. A bounded read prevents
        # unbounded temporary disk use even when object metadata is incorrect.
        try:
            with fsspec.open(
                url, "rb", **s3_options(endpoint_url, anonymous=s3_anonymous)
            ) as stream:
                _copy_bounded(stream, destination, maximum)
        except (OSError, ValueError):
            raise
        except Exception as error:
            raise OSError("Cannot read remote S3 object; check credentials and endpoint") from error
        return
    try:
        request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "twhy"})
        with urlopen(request, timeout=20) as response:
            final = urlsplit(response.geturl())
            if final.scheme not in {"http", "https"}:
                raise ValueError("HTTP redirects must stay within HTTP(S)")
            advertised = response.headers.get("Content-Length")
            if advertised is not None and advertised.isdecimal() and int(advertised) > maximum:
                raise ValueError(
                    "Remote file exceeds --max-remote-mib; increase the limit explicitly"
                )
            if (response.headers.get("Content-Encoding", "identity").casefold()
                    not in {"", "identity"}):
                raise ValueError("The server returned encoded HTTP content instead of file bytes")
            _copy_bounded(response, destination, maximum)
    except HTTPError as error:
        raise OSError(f"HTTP download failed with status {error.code}") from error
    except URLError as error:
        raise OSError("HTTP download failed; check the address and connection") from error


@contextmanager
def resolve_input(source: str | Path, *, max_remote_mib: float = DEFAULT_MAX_REMOTE_MIB,
                  s3_endpoint_url: str | None = None, s3_anonymous: bool = False,
                  input_format: str = "auto") -> Iterator[ResolvedInput]:
    """Yield a stable input. Remote Zarr remains lazy; bounded files use a temporary copy.

    Only the particular staged remote file is deleted on exit. Local inputs and
    remote Zarr are never changed. This context must surround the entire viewer,
    including its spawned PDF refinement workers and selection activity.
    """
    if not isfinite(max_remote_mib) or max_remote_mib <= 0:
        raise ValueError("max_remote_mib must be positive and finite")
    maximum = int(max_remote_mib * _BLOCK)
    if maximum < 1:
        raise ValueError("max_remote_mib must permit at least one byte")
    if s3_endpoint_url is not None:
        s3_options(s3_endpoint_url)  # Fail clearly even when the input is not S3.
    zarr = is_zarr_input(source, input_format=input_format)
    scheme = remote_scheme(source)
    label = display_name(source)
    if scheme:
        url = validate_url(source, zarr=zarr)
        if zarr:
            yield ResolvedInput(url, label, True, True)
            return
        extension = Path(urlsplit(url).path).suffix.lower()
        # Preserve only the suffix needed for validated Aperio routing. A filename
        # (or signed query) is never used as a filesystem path or log message.
        suffix = extension if re.fullmatch(r"\.[a-z0-9]{1,10}", extension) else ".bin"
        with TemporaryDirectory(prefix="twhy-input-") as directory:
            path = Path(directory) / ("download" + suffix)
            _download(url, path, maximum, endpoint_url=s3_endpoint_url,
                      s3_anonymous=s3_anonymous)
            yield ResolvedInput(path, label, False, True)
        return
    path = Path(source).expanduser()
    if zarr:
        if not path.is_dir():
            raise FileNotFoundError(f"OME-Zarr store directory does not exist: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    yield ResolvedInput(path, label, zarr, False)
