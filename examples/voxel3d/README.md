# voxel3d: multilayer TiO2/air voxel colour router

Inverse design of a 2 um x 2 um colour-router cell made of independently patterned TiO2/air voxel layers
(glass superstrate -> voxel layers -> air gap -> detector plane, 4-well RGGB readout).

- `src/voxel3d/` multilayer RCWA forward (one layer eigenproblem per layer, polarisation shared, Redheffer star product),
  with chunked recomputation in the backward pass so that 50-100 layers fit in memory; layer-eigenproblem and geometry helpers;
  tabulated Siefke ALD TiO2 index.
- Connectivity gate (in the drivers): each layer is supported by the layer below it (grey: `rho_i * clamp(2*dilate(rho_{i-1}), 0, 1)`,
  binary: exact dilation, checked independently with `torch.roll`), so no floating structures. Layer 0 rests on the glass.
- `examples/voxel3d/run_classic.py` and `run_classic10nm.py`: peak x purity objective per colour band (bands 420/450/470, 510/540/570,
  600/635/670 nm), diagonal (x<->y) symmetry. `run_classic10nm.py` is the 10 nm-layer variant (single normal ray by default,
  3D Gaussian control filter, `lr = min(lr, (lrbeta / lr-h) / beta)`; beta is multiplied by `--bstep` when the 3-step change of the
  grey figure of merit is saturated (between -1 % and +0.5 %), at least `--smin` steps after the previous advance, up to `--betamax`;
  the run stops when the 10-step grey rise is below 0.05 % and the grey-to-binary gap at the same step is at most 0.1 %).
- `examples/voxel3d/run_voxel.py`: imaging-information (`I_img`) objective. It needs the matched-context module
  `cr_itd_v2.optimization`, which is not part of this repository.
- `examples/voxel3d/test_voxel_parity.py`: CPU parity of the multilayer forward against the reference `torcwa` forward
  (values, gradients, finite differences).

## Requirements

Python >= 3.10, `numpy`, `torch`, `torcwa` (0.1.4.2); a CUDA GPU for the optimisation runs. `I_img` is logged as NaN by the
classic drivers when the matched context is not available.

## Usage

```bash
pip install -e .            # from the repository root
python examples/voxel3d/test_voxel_parity.py --layers 3      # CPU

# peak x purity, 10 layers x 100 nm, 16-ray pupil ensemble
python examples/voxel3d/run_classic.py --out runs/cls10 --device cuda

# peak x purity, 100 layers x 10 nm, normal incidence
python examples/voxel3d/run_classic10nm.py --out runs/cls10nm --layers 100 --h 0.01 --sigma-nm 50 --chunk 3
```

Key options: `--layers`, `--h` (layer height, um), `--gap-um`, `--rays`, `--sigma-nm`, `--beta0`, `--betamax`, `--bstep`,
`--lr`, `--lrbeta`, `--lr-h`, `--bineval` (binary evaluation interval), `--chunk` (layers per backward chunk), `--init-state`,
`--lr-ramp-steps` (geometric lr ramp from the beta-rule value to `--lr`), `--gray-pen` / `--gray-ramp` (grayness penalty
`lam * mean(4 rho (1 - rho))` and its linear ramp), `--adam-restart` (Adam bias correction from the launch step).
Each step writes `steps/*.npz` and `history.json`; `state.pt` allows resuming (rerun the same command). Create `<out>/STOP` to
stop cleanly.
