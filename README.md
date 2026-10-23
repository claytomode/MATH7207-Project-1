# 2D pose-graph SLAM

MATH 7207 mini-project. Maximum-likelihood estimation of SE(2) poses, refined by Levenberg–Marquardt from a dead-reckoning start and from a spectral start.

```
julia --project=. demo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## Figures

`demo.jl` rewrites `figures/`. The copies in the repo match `figures/results.txt`. Dashed is the start, solid is after Levenberg–Marquardt, black is the truth, and gray is a loop closure.

- `noiseless_paths.png`, `noiseless_cost.png`: exact measurements. Both starts recover the loop.
- `moderate_paths.png`, `moderate_cost.png`: one loop closure. Both starts finish on the same path.
- `hard_grid_paths.png`, `hard_grid_cost.png`: large heading noise. Dead reckoning stops in the wrong valley.
- `spiral_paths.png`, `spiral_cost.png`: the same split on a spiral.

## Sources

Levenberg and Marquardt: standard LM solver.
Grisetti, Kümmerle, Stachniss, and Burgard (2010): the dead-reckoning start.
Singer (2011) and Rosen, Carlone, Bandeira, and Leonard, SE-Sync (2019): the spectral start. This repository implements that start but doesn't run SE-Sync.
