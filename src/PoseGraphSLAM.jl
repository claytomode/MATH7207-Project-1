"""
2D pose-graph SLAM (MATH 7207).

A robot moves in the plane. Each pose is an element of SE(2): a position (x, y)
and a heading. Edges carry a noisy measurement of the relative pose between two
poses, plus a positive information weight. The maximum-likelihood estimate
minimizes

    Σ  || L_ij * Log( Z_ij⁻¹ X_i⁻¹ X_j ) ||²

Levenberg–Marquardt refines a dead-reckoning start and a spectral start. The
first pose stays at the identity. Synthetic loops and a grid supply the
comparisons, with the ground truth known.
"""
module PoseGraphSLAM

using LinearAlgebra
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
export noiseless_graph, moderate_graph, grid_graph, hard_graph

end
