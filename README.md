# 2D pose-graph SLAM

MATH 7207 mini-project. Maximum-likelihood estimation of SE(2) poses, refined by Levenberg–Marquardt from a dead-reckoning start and from a spectral start.

```
julia --project=. demo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```
