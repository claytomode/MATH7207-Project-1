# 2D pose-graph SLAM

MATH 7207 mini-project. Maximum-likelihood estimation of SE(2) poses, refined by Levenberg–Marquardt from a dead-reckoning start and from a spectral start.

```
julia --project=. demo.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

## Sources

**Cost and update.** For an edge from pose i to pose j the residual is Log(Z_ij^{-1} X_i^{-1} X_j). A step updates each free pose on the right, X ← X ∘ Exp(δ). Pose 1 is fixed at the identity. This is the pose-graph factor used by g2o and GTSAM.

- Kümmerle, Grisetti, Strasdat, Konolige, Burgard. g2o: A general framework for graph optimization. ICRA, 2011.
- Dellaert, Kaess. Factor Graphs for Robot Perception. Foundations and Trends in Robotics, 2017.
- Solà, Deray, Atchuthan. A micro Lie theory for state estimation in robotics. arXiv:1812.01537, 2018.

**Levenberg–Marquardt.** Each iteration solves (JᵀJ + λI) δ = −Jᵀr. A step that lowers the cost is kept and λ shrinks. Otherwise the step is rejected and λ grows.

- Levenberg. A method for the solution of certain non-linear problems in least squares. Quarterly of Applied Mathematics, 1944.
- Marquardt. An algorithm for least-squares estimation of nonlinear parameters. Journal of the Society for Industrial and Applied Mathematics, 1963.

**Dead reckoning.** Chain the odometry edges, X_j = X_i ∘ Z_ij, and ignore loop closures. This is the usual odometry initialization in graph-based SLAM.

- Grisetti, Kümmerle, Stachniss, Burgard. A tutorial on graph-based SLAM. IEEE Intelligent Transportation Systems Magazine, 2010.

**Spectral start.** Build the rotational connection Laplacian, take the two eigenvectors for the smallest eigenvalues, project each 2×2 block onto SO(2), then solve a linear least-squares problem for the translations. That initialization is the one used by SE-Sync. The rotation step is angular synchronization.

- Singer. Angular synchronization by eigenvectors and semidefinite programming. Applied and Computational Harmonic Analysis, 2011.
- Rosen, Carlone, Bandeira, Leonard. SE-Sync: A certifiably correct algorithm for synchronization over the special Euclidean group. International Journal of Robotics Research, 2019.
- Carlone, Tron, Daniilidis, Dellaert. Initialization techniques for 3D SLAM: a survey on rotation estimation and its use in pose graph optimization. ICRA, 2015.
