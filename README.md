# 2D pose-graph SLAM

MATH 7207 mini-project. Maximum-likelihood estimation of SE(2) poses, refined by Levenberg–Marquardt from a dead-reckoning start and from a spectral start.

```
julia --project=. demo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## Sources

Levenberg and Marquardt: standard LM solver.
Grisetti, Kümmerle, Stachniss, and Burgard (2010): the dead-reckoning start.
Singer (2011) and Rosen, Carlone, Bandeira, and Leonard, SE-Sync (2019): the spectral start. This repository implements that start but doesn't run SE-Sync.
