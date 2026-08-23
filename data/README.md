# Figure-reproduction data

This directory holds the final design masks and the derived response caches
needed to reproduce the figures in the paper. Nothing here requires an RCWA
solve to read; every published number is recomputable from the cached arrays.

## `masks/`

Boolean design masks on a 128 x 128 grid over a 2.0 x 2.0 um period
(`True` = high-index pillar material, SiN on an SiO2 substrate). See
`manifest.json` for each mask's field slot, chief-ray angle/azimuth, seed,
and height.

| file | role | figure |
|------|------|--------|
| `r1a.npy` .. `r3c.npy` | six per-slot deployed designs | six-slot / sector figures |
| `champion_r1a.npy` | champion pupil-ensemble router carried across the field | main router figures |

## `caches/`

- `six_slot_spectra.npz` / `six_slot_table.json` — per-slot (well, wavelength)
  responses and the summary table for the six-slot figure.
- `imaging_demo_cache.npz` — ground-truth and reconstructed image panels with
  PSNR for the imaging demonstration.
- `field_map.npz` — router vs. filter-array score over the (theta, phi) field grid.
- `fig1_crossover.npz` — SNR sweep with the CR/CFA crossover point (Fig. 3).
- `spatial_kernel_r1a.npz`, `_r2a.npz`, `_r3b.npz` — per-slot spatial kernels,
  MTF/LSF/ESF and spill matrices.
- `cold_pair_table.json` + `paired_control_cache/` — the paired
  pupil-vs-plane control comparison (two seeds). `responses.npz` keys are
  `<design_id>@<slot>`; `masks/<design_id>.npy` is the design each row scored.
  Design ids are `pupil_seed<seed>` / `plane_seed<seed>`.

Response array axes: rows are wells in the order given by `__well_order__`
(R, G2, G1, B); columns are wavelengths in nm
(420, 450, 470, 510, 540, 570, 600, 635, 670).
