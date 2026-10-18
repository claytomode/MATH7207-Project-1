"""
2D pose-graph SLAM (MATH 7207).

A robot moves in the plane. Each pose is an element of SE(2): a position (x, y)
and a heading. Edges carry a noisy measurement of the relative pose between two
poses, plus a positive information weight. The maximum-likelihood estimate
minimizes

    Σ  || L_ij * Log( Z_ij⁻¹ X_i⁻¹ X_j ) ||²

where Log turns a relative pose into a 3-vector (translation in the Lie algebra,
and heading). L_ij is a diagonal square root of the information. When the three
diagonal weights are equal to κ_ij, this is κ_ij times the squared length of the
log, which is the isotropic case in the proposal. Inverse-variance weights are
the maximum-likelihood choice when translation (meters) and heading (radians)
have different noise.

Algorithm
1. Build the measurement graph from the edges.
2. Two starts. Dead reckoning chains the odometry edges and ignores loop
   closures. The spectral start builds the rotational connection Laplacian,
   takes the two eigenvectors for the smallest eigenvalues, projects each 2×2
   block onto SO(2), and recovers translations by linear least squares. That
   pair (R, t) is a pose in SE(2).
3. Evaluate the pose-graph cost above.
4. Refine with Levenberg–Marquardt. Each iteration solves
   (JᵀJ + λ I) δ = −Jᵀr and updates X ← X ∘ Exp(δ). If the cost drops, accept
   the step and shrink λ. If not, reject it and grow λ.
5. Hold the first pose at the identity so the map cannot slide or spin. Stop
   when ||Jᵀr|| is small, or when the iteration limit is hit.
6. Run the refinement from both starts and compare cost, iterations, and pose
   error against the ground truth after the gauge is fixed.

A disconnected graph has no solution once the first pose is fixed. `solve`
throws `DisconnectedGraphError` instead of returning a meaningless estimate.
"""
module PoseGraphSLAM

using LinearAlgebra
using Printf
using Random

include("se2.jl")
include("problem.jl")
include("initialize.jl")
include("optimize.jl")
include("synthetic.jl")

export Pose, Edge, PoseGraph, PoseError, SolveResult, Comparison
export IDENTITY, DisconnectedGraphError
export compose, inverse, between, exp_se2, log_se2, wrap_angle, align_gauge
export pose_cost, residuals, residual_and_jacobian, jacobian_finite_difference
export jacobian_discrepancy, pose_error, disconnected_poses
export dead_reckoning, spectral_initialization, levenberg_marquardt, solve, compare, init_label
export loop_graph, toy_graph, toy_evaluation_state, disconnected_graph
export noiseless_graph, moderate_graph, grid_graph, hard_graph, maze_graph, crazy_maze

end
