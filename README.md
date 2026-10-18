# colored-inkt

Colour filters for looking at ink in CT scans of the Herculaneum scrolls.

You give it a surface volume (the stack of layers around a traced papyrus segment). It finds the papyrus face
pixel by pixel, flattens the layers onto it, and turns what it sees there into a colour image. The filters were
tuned on the [Vesuvius Challenge](https://scrollprize.org) open data, where ink labels exist, and judged by eye.

This is a set of display filters, not an ink detector. Nothing here is trained and nothing says "this is ink".
The filters make some properties of the surface easier to see, and ink often has those properties. Whether a mark
is a letter is still for a person to judge, so always compare with the plain CT (the `mid_slice` filter).

The package has two parts:

- `colored-inkt render`: a command line tool that writes one PNG per filter for a segment of any size.
- `colored-inkt serve`: a small web viewer that runs on your own computer. You can add your own segments, try
  every filter, change their colours and settings live, and step through depths.

## The three recommended filters

| Name | What it shows |
|---|---|
| `two_hue` | Two ink signatures as two colours on top of the unchanged surface. Extra density at the face is tinted amber. A rough, unoriented surface (crackle) is tinted blue. Where both occur the colours mix towards grey-violet. Specks up to 50 µm are removed first. |
| `isoluminant` | The surface stays exactly as bright as it is, and the ink cue only adds colour (amber) at the same brightness. The texture you would read in the plain CT is not changed. |
| `lbp` | Texture contrast measured on a small circle around each pixel, in any direction (rotation-invariant local binary pattern variance). Ink-covered surface is often smoother or rougher than the open fibre weave around it. |

There are 79 filters in total. `colored-inkt filters --all` lists them, grouped by family, and the viewer's
Methods tab explains each one. Most of them are earlier attempts and baselines, kept for comparison.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/brkzak/inkt-filter
cd inkt-filter
pip install .
```

or, with uv, `uv tool install .` (or `uv run colored-inkt ...` from the folder).

Three of the 79 filters (ICA, k-means and non-local means) also need scikit-learn and OpenCV:
`pip install ".[extra]"`. The three recommended filters do not.

## Command line

```bash
colored-inkt render SOURCE -o OUTPUT_FOLDER [options]
```

`SOURCE` can be:

- a folder of layer images (`00.tif`, `01.tif` and so on, the `layers/` folder of a published segment),
  read in file name order;
- one multi-page TIFF or a `.npy` file with the whole stack (layers, height, width);
- a Zarr or OME-Zarr, as a local path or an https URL, for example a `surface-volumes/*.zarr` of the open data.

8-bit and 16-bit data both work.

Some examples:

```bash
# A 1024 px square of an open-data segment, read straight from the bucket at pyramid level 2 (about 9.6 µm
# pixels). The voxel size is in the OME-Zarr metadata. This takes about 40 seconds on a laptop.
colored-inkt render https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr \
    --level 2 --box 160 192 1024 1024 -o w041

# A layers folder from a 7.91 µm scan. The layers are resampled to 2.4 µm planes first.
colored-inkt render segment/layers --voxel-um 7.91 -o out

# A large segment at full resolution, binned 2 × 2 in-plane, with the ink on the other face.
colored-inkt render segment/layers --voxel-um 2.4 --bin 2 --side -1 -o out

# Every filter, with a quarter-size JPEG of each next to the PNG.
colored-inkt render stack.tif --voxel-um 2.4 -f all --preview -o out
```

The output folder gets `two_hue.png`, `isoluminant.png`, `lbp.png` (or the filters you asked for with `-f`) and
`render.json`, which records the source, box and every setting. Pixels with no data are black.

| Option | Meaning |
|---|---|
| `-f NAME ...` | Filters by name (`two_hue` or the full `gen_two_hue`), or `all`. |
| `--voxel-um` | Voxel size in µm. Needed unless an OME-Zarr says it. Use `--layer-um` and `--pixel-um` if they differ. |
| `--level` | Pyramid level of a Zarr. |
| `--box Y X H W` | Only this box, in pixels of the source (at that level). |
| `--bin N` | Average N × N pixels in-plane before anything else. |
| `--side 1/-1` | Which way the inked face looks: towards higher layer numbers (1, the default) or lower (-1). |
| `--notch`, `--despeckle` | Pre-filters for the filters that do not set their own (see below). |

Large areas are rendered in 2048 px tiles that overlap by 256 px and are blended, so there are no seams. The
intermediate results are kept on disk next to the output (`OUTPUT_FOLDER/.colored-inkt-work`, or `--work`)
and removed at the end. They take about 80 bytes per pixel of the area, so a 10000 × 10000 px segment needs
about 8 GB of free disk for a while.

## The viewer

```bash
colored-inkt serve
```

Then open <http://127.0.0.1:6544>. Segments and renders are kept in `~/.colored-inkt` (change it with `--data`).

The viewer has two tabs.

**Segments** is where you look at a segment through a filter. On the left is the list of segments. It starts with
ten open-data examples from different scrolls, each a labelled square of about 2 × 2 cm. Nothing is downloaded
until you open one and press "Download and prepare", which takes a minute or two. For the examples you can switch
on the ink label outline to see where people marked ink.

Pick a filter, and the image is computed the first time (this can take a minute for some filters; later it is
instant). Filters that are recoloured in the browser show sliders for their colours and ranges. You can compare
two filters (swipe, blink or side by side), star the ones you like, and scan depth: the same filter rendered as if the segment were
a few planes in front of or behind the face it found.

**Methods** lists every filter with a short explanation of what it computes.

### Adding your own segment

Press "+ add a segment". There are two ways in:

- **Upload files.** Choose the layer images (one file per layer, or one multi-page TIFF or `.npy`), and an ink
  label image of the same size if you have one. The files are copied into the data folder.
- **Folder, file or URL.** Give a path on the computer that runs the server, or an https URL of a Zarr, and the
  pyramid level. Nothing is uploaded; it is read from there.

Then give the layer spacing and pixel size in µm (an OME-Zarr fills these in itself), which face the ink is on,
optional binning, and a crop. The viewer takes up to 4096 px per side after binning; without a crop it takes the
centre 2048 px square. For anything larger, use the command line.

If the colours look like the inside of the sheet rather than its face, the ink is probably on the other side:
use the flip button and the segment is prepared again.

## How it works, briefly

1. **Reading.** The stack is read in 512 px blocks with a 48 px border of context. Layers at another spacing than
   2.4 µm (more than 10 % off) are resampled linearly to 2.4 µm planes centred on the middle layer, because the
   filters measure depth in 2.4 µm planes. Pixels where every layer is 0 are marked as empty and left out.
2. **Finding the face.** For each pixel the papyrus face is found where the density falls most steeply when
   going from the sheet into the air or gap in front of the inked face (the direction `--side` gives). A coarse
   search over a smoothed stack comes first, then a finer one around it. The layers are then sampled at fixed distances from
   that face (from 8 planes outside to 16 planes inside), which flattens the surface.
3. **Filtering.** Each filter reads the flattened stack (often just the first few planes inside the face, the
   "skin") and computes one or more cues, such as extra density, texture contrast or roughness.
4. **Colours.** The display range of each cue is set once per area from the area itself, without any labels
   (like window and level in a CT viewer). This keeps an empty stretch of papyrus from being stretched into
   false structure. For a large area the range comes from a grid of nine 1024 px squares.

Two optional pre-filters are applied before a filter runs. The despeckle filter removes small dense specks
(30, 50 or 90 µm), which otherwise look like ink to density cues. The notch filter takes the regular fibre
pattern out of the image by blanking a narrow wedge of directions (± 4, 7 or 12°) in its Fourier transform.
`two_hue` always uses despeckle at 50 µm.

## Limitations

- The filters were tuned at 2.4 µm on the open-data segments. At the 8 to 9 µm of whole-scroll scans they run
  (the layers are resampled), but how well they show ink there has not been checked against labels.
- If the face search picks the wrong surface (a gap between sheets, a fold), every filter shows the wrong layer.
  The depth scan in the viewer helps to see this.
- The viewer holds one segment's data in memory, up to a few GB for a 4096 px square.
- The viewer has no login. It listens on 127.0.0.1 (this computer only) by default. If you start it with
  `--host 0.0.0.0`, anyone who can reach that port can see your segments and add files.

## Tests

```bash
pip install ".[dev,extra]"
pytest
```

The tests run on small synthetic stacks and need no network or GPU.

## Credits and licence

The code is MIT licensed (see `LICENSE`). It was developed by Brian (brkzak), with the help of an AI coding
assistant, in a research fork of [ScrollPrize/villa](https://github.com/ScrollPrize/villa) (MIT).

The example segments, ink labels and supervision masks are Vesuvius Challenge open data
(<https://scrollprize.org/data>). They are not included here; the viewer downloads them from the public bucket
when you prepare one, and their use is subject to the challenge's data terms.
