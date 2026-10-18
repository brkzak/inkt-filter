"""Reading surface volumes: the (layers, height, width) CT stack around a traced segment.

Three kinds of input are accepted, local or (for Zarr) over HTTP:

* a folder of layer images, one 2D image per layer (``00.tif`` … ``64.tif``, the usual layout of a published
  segment's ``layers/`` folder), taken in file-name order;
* one multi-page TIFF (or a ``.npy`` file) holding the whole stack;
* a Zarr array or OME-Zarr pyramid (the open-data ``surface-volumes/*.zarr``), with the pyramid level to read.

Every source is read lazily: ``read(y, x, h, w)`` returns only that window of every layer, padded with zeros
outside the image, so large segments never have to fit in memory. 16-bit data is handed on as 16-bit; the filters
scale it to the 8-bit range themselves (``Volume.value_scale``).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAYER_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class Volume:
    """A lazily read stack. ``reader(y0, y1, x0, x1)`` returns (layers, y1 - y0, x1 - x0) inside the bounds."""

    shape: tuple[int, int, int]  # (layers, height, width)
    dtype: np.dtype
    reader: object
    description: str

    @property
    def value_scale(self) -> float:
        """Factor that brings the values to the 0 … 255 range the filters were tuned on."""
        return 1.0 / 257.0 if np.dtype(self.dtype).itemsize == 2 else 1.0

    def read(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        """(layers, h, w) at (y, x), zero outside the image."""
        layers, height, width = self.shape
        out = np.zeros((layers, h, w), dtype=self.dtype)
        y0, y1 = max(0, y), min(height, y + h)
        x0, x1 = max(0, x), min(width, x + w)
        if y1 > y0 and x1 > x0:
            out[:, y0 - y : y1 - y, x0 - x : x1 - x] = self.reader(y0, y1, x0, x1)
        return out


def open_volume(source: str | Path | list, *, level: int = 0) -> Volume:
    """A ``Volume`` for a layer folder, a list of layer files, a multi-page TIFF, a ``.npy`` file or a Zarr.

    Raises:
        FileNotFoundError: when a local path does not exist.
        ValueError: when the input is not one of the supported kinds, or the layers differ in size.
    """
    if isinstance(source, (list, tuple)):
        return _layer_files([Path(p) for p in source])
    text = str(source)
    if text.startswith(("http://", "https://")) or text.rstrip("/").endswith(".zarr") or _is_zarr(Path(text)):
        return _zarr(text, level)
    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.is_dir():
        files = sorted((p for p in path.iterdir() if p.suffix.lower() in LAYER_SUFFIXES), key=_natural)
        if not files:
            raise ValueError(f"{path} holds no layer images ({', '.join(LAYER_SUFFIXES)})")
        return _layer_files(files)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        return _array(array, f"{path.name}")
    if path.suffix.lower() in (".tif", ".tiff"):
        return _multipage(path)
    raise ValueError(f"{path}: expected a folder of layers, a .tif, a .npy or a .zarr")


def _natural(path: Path):
    """Sort key so that 2.tif comes before 10.tif."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _is_zarr(path: Path) -> bool:
    return path.is_dir() and any((path / f).exists() for f in (".zarray", ".zgroup", "zarr.json"))


def _array(array, description: str) -> Volume:
    if array.ndim != 3:
        raise ValueError(f"{description}: expected a 3D stack (layers, height, width), got shape {array.shape}")
    return Volume(tuple(array.shape), array.dtype, lambda y0, y1, x0, x1: np.asarray(array[:, y0:y1, x0:x1]),
                  description)


def _zarr(url: str, level: int) -> Volume:
    """One array of a Zarr store; a group (OME-Zarr pyramid) is read at ``level``."""
    import zarr

    node = zarr.open(url, mode="r")
    if isinstance(node, zarr.Group):
        node = node[str(level)]
    return _array(node, f"{url} (level {level})")


def _multipage(path: Path) -> Volume:
    import tifffile

    try:
        array = tifffile.memmap(path, mode="r")
    except ValueError:  # compressed or tiled: read it once
        array = tifffile.imread(path)
    return _array(array, path.name)


def _layer_files(files: list[Path]) -> Volume:
    """Layers from one 2D image each (uncompressed TIFFs are memory-mapped, anything else is decoded once)."""
    if not files:
        raise ValueError("no layer files given")
    layers = [_layer(p) for p in files]
    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise ValueError(f"the layers differ in size: {sorted(shapes)}")
    if layers[0].ndim != 2:
        raise ValueError(f"{files[0].name}: expected a 2D gray image, got shape {layers[0].shape}")
    dtype = layers[0].dtype

    def reader(y0, y1, x0, x1):
        return np.stack([np.asarray(layer[y0:y1, x0:x1]) for layer in layers])

    return Volume((len(layers), *layers[0].shape), dtype, reader, f"{len(files)} layers in {files[0].parent}")


@functools.lru_cache(maxsize=None)
def _decoded(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    return array if array.ndim == 2 else array[..., :3].mean(axis=-1).astype(np.uint8)


def _layer(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        try:
            return tifffile.memmap(path, mode="r")
        except ValueError:
            return tifffile.imread(path)
    return _decoded(path)


def read_label(source: str | Path, level: int = 0) -> np.ndarray | object:
    """A 2D ink label (nonzero = ink): an image file, or a Zarr/OME-Zarr read lazily (sliceable)."""
    text = str(source)
    if text.startswith(("http://", "https://")) or text.rstrip("/").endswith(".zarr"):
        import zarr

        node = zarr.open(text, mode="r")
        return node[str(level)] if isinstance(node, zarr.Group) else node
    path = Path(text).expanduser()
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        array = tifffile.imread(path)
    else:
        array = np.asarray(_decoded(path))
    if array.ndim == 3:
        array = array.max(axis=-1) if array.shape[-1] <= 4 else array.max(axis=0)
    return array


def ome_voxel_um(source: str | Path, level: int = 0) -> tuple[float, float] | None:
    """(layer µm, pixel µm) of an OME-Zarr pyramid level from its metadata, or None if it does not say."""
    text = str(source)
    if not (text.startswith(("http://", "https://")) or text.rstrip("/").endswith(".zarr") or _is_zarr(Path(text))):
        return None
    import zarr

    try:
        node = zarr.open(text, mode="r")
    except Exception:
        return None
    if not isinstance(node, zarr.Group):
        return None
    attrs = dict(node.attrs)
    multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
    if not multiscales:
        return None
    for dataset in multiscales[0].get("datasets", []):
        if str(dataset.get("path")) != str(level):
            continue
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale" and len(transform.get("scale", [])) == 3:
                z, y, x = (float(v) for v in transform["scale"])
                return z, (y + x) / 2
    return None
