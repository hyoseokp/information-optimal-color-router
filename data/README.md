# Data

Design masks and the source data of the main-text figures.

## `masks/`

Boolean design masks on a 128 x 128 grid over a 2.0 x 2.0 um period (`True` = SiN pillar, 800 nm tall, on SiO2).
`manifest.json` gives each mask's design name, field sector (chief-ray angle and azimuth) and imaging information.

| file | design | chief-ray angle, azimuth |
|------|--------|--------------------------|
| `r1a.npy` | M1 | 2.5°, 22.5° |
| `r2a.npy` | M2 | 10°, 11.25° |
| `r2b.npy` | M3 | 10°, 33.75° |
| `r3a.npy` | M4 | 20°, 7.5° |
| `r3b.npy` | M5 | 20°, 22.5° |
| `r3c.npy` | M6 | 20°, 37.5° |

## `source/`

Plotted values of the main-text figures. Each JSON file starts with a `description` field.

| file | figure |
|------|--------|
| `fig1c_designs.json` | Fig. 1c (and the Supplementary low-light figure): imaging information of the plane-wave-optimized and pupil-optimized designs, M1 and the four literature routers |
| `fig2_sectors.json` | Fig. 2: imaging information at the six sector centers, area-weighted values, site responses |
| `fig3_crossover.json` | Fig. 3: information difference to the CFA against the green-site SNR, crossovers, read-noise dependence |
| `fig4_purity.json` | Fig. 4: purity and throughput of M1 over the pupil directions |
| `fig5_measured.json` | Fig. 5d,e and Fig. 1e: site fractions and information difference of the fabricated M1, measured and simulated |

Imaging information is in bit per raw pixel. Sites q1-q4 follow the labels of the figures.

## `measured/`

Raw camera frames of the fabricated M1, one per wavelength from 400 to 700 nm in 25 nm steps (`raw/M1_<wavelength>nm.npz`,
key `frame`, 2048 x 2048 uint16 counts as recorded). `frames.json` lists the exposure time of each frame, the pixel size at
the sample and the camera dark offset.
