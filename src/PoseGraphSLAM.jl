"""
2D pose-graph SLAM (MATH 7207).

A robot moves in the plane. Each pose is an element of SE(2): a position (x, y)
and a heading. Edges carry a noisy measurement of the relative pose between two
poses, plus a positive information weight. The maximum-likelihood estimate
minimizes

    Σ  || L_ij * Log( Z_ij⁻¹ X_i⁻¹ X_j ) ||²

Two starts are available. Dead reckoning chains the odometry edges and ignores
loop closures. The spectral start builds the rotational connection Laplacian,
takes the two eigenvectors for the smallest eigenvalues, projects each 2×2
block onto SO(2), and recovers translations by linear least squares.
"""
module PoseGraphSLAM

using LinearAlgebra

include("se2.jl")
include("problem.jl")
include("initialize.jl")

export Pose, Edge, PoseGraph, PoseError, IDENTITY, DisconnectedGraphError
export compose, inverse, between, exp_se2, log_se2, wrap_angle, align_gauge
export pose_cost, residuals, residual_and_jacobian, jacobian_finite_difference
export jacobian_discrepancy, pose_error, disconnected_poses
export dead_reckoning, spectral_initialization

end
