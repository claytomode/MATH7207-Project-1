"""
2D pose-graph SLAM (MATH 7207).

Poses are elements of SE(2): a position (x, y) and a heading. The cost and the
solver are built on the group operations in this module.
"""
module PoseGraphSLAM

using LinearAlgebra

include("se2.jl")

export Pose, IDENTITY
export compose, inverse, between, exp_se2, log_se2, wrap_angle, align_gauge

end
