"""
2D pose-graph SLAM (MATH 7207).

A robot moves in the plane. Each pose is an element of SE(2): a position (x, y)
and a heading. Edges carry a noisy measurement of the relative pose between two
poses, plus a positive information weight. The maximum-likelihood estimate
minimizes

    Σ  || L_ij * Log( Z_ij⁻¹ X_i⁻¹ X_j ) ||²

where Log turns a relative pose into a 3-vector. L_ij is a diagonal square root
of the information. When the three diagonal weights are equal to κ_ij, this is
κ_ij times the squared length of the log.
"""
module PoseGraphSLAM

using LinearAlgebra

include("se2.jl")
include("problem.jl")

export Pose, Edge, PoseGraph, PoseError, IDENTITY, DisconnectedGraphError
export compose, inverse, between, exp_se2, log_se2, wrap_angle, align_gauge
export pose_cost, residuals, residual_and_jacobian, jacobian_finite_difference
export jacobian_discrepancy, pose_error, disconnected_poses

end
