using Test
using PoseGraphSLAM
using LinearAlgebra

function _close(a::Pose, b::Pose; atol = 1e-8)
    return abs(a.x - b.x) <= atol && abs(a.y - b.y) <= atol && abs(wrap_angle(a.θ - b.θ)) <= atol
end

@testset "SE(2) exp and log" begin
    for ξ in ([0.0, 0.0, 0.0], [0.4, -0.2, 0.0], [1.5, 0.3, 0.7], [-0.8, 2.0, -2.4], [0.0, 0.0, π - 1e-4])
        pose = exp_se2(ξ)
        recovered = log_se2(pose)
        @test recovered ≈ [ξ[1], ξ[2], wrap_angle(ξ[3])] atol = 1e-8
    end
    pose = Pose(1.3, -0.7, 1.1)
    @test _close(exp_se2(log_se2(pose)), pose)
    @test _close(compose(inverse(pose), pose), IDENTITY)
    @test _close(compose(pose, inverse(pose)), IDENTITY)
end

@testset "adjoint" begin
    pose = Pose(1.2, -0.7, 0.4)
    ξ = [0.3, -0.2, -0.5]
    moved = exp_se2(PoseGraphSLAM.Ad(pose) * ξ)
    conjugated = compose(pose, compose(exp_se2(ξ), inverse(pose)))
    @test _close(moved, conjugated; atol = 1e-8)
end

@testset "right Jacobian inverse" begin
    φ = [0.4, -0.25, 0.6]
    E = exp_se2(φ)
    ε = 1e-6
    numerical = zeros(3, 3)
    for k in 1:3
        η = zeros(3)
        η[k] = 1.0
        plus = log_se2(compose(E, exp_se2(ε * η)))
        minus = log_se2(compose(E, exp_se2(-ε * η)))
        numerical[:, k] = (plus - minus) / (2ε)
    end
    @test PoseGraphSLAM.right_jacobian_inverse(φ) ≈ numerical atol = 1e-7
end

@testset "toy graph residual and Jacobian" begin
    graph = toy_graph()
    @test graph.n == 3
    @test length(graph.edges) == 3
    # At the ground truth the measurements are exact, so every residual is zero.
    @test norm(residuals(graph, graph.truth)) < 1e-10
    poses = toy_evaluation_state(graph)
    gap = jacobian_discrepancy(graph, poses; ε = 1e-6)
    @test gap < 1e-6
end

@testset "noiseless loop" begin
    graph = noiseless_graph(; n = 12, radius = 8.0)
    for init in (:dead_reckoning, :spectral)
        result = solve(graph; init = init, grad_tol = 1e-12)
        err = pose_error(result.poses, graph.truth)
        @test result.cost < 1e-8
        @test err.pos_max < 1e-5
        @test err.heading_max < 1e-5
    end
    # A nearby start, not an exact one, still has to fall back to the truth.
    poses0 = copy(graph.truth)
    for i in 2:graph.n
        poses0[i] = compose(graph.truth[i], exp_se2([0.05, -0.04, 0.03]))
    end
    refined = levenberg_marquardt(graph, poses0; grad_tol = 1e-12, max_iters = 25)
    err = pose_error(refined.poses, graph.truth)
    @test refined.cost < 1e-8
    @test err.pos_max < 1e-5
    @test err.heading_max < 1e-5
end

@testset "moderate loop and hard grid" begin
    moderate = moderate_graph()
    moderate_runs = compare(moderate; max_iters = 40)
    for result in (moderate_runs.dead_reckoning, moderate_runs.spectral)
        err = pose_error(result.poses, moderate.truth)
        @test err.pos_rmse < 1.0
        @test err.heading_rmse < 0.15
    end

    hard = hard_graph()
    hard_runs = compare(hard; max_iters = 80)
    dr_err = pose_error(hard_runs.dead_reckoning.poses, hard.truth)
    sp_err = pose_error(hard_runs.spectral.poses, hard.truth)
    @test sp_err.pos_rmse < 0.5
    @test dr_err.pos_rmse > 5
    @test hard_runs.dead_reckoning.cost > 10 * hard_runs.spectral.cost
end

@testset "disconnected graph" begin
    graph = disconnected_graph()
    @test !isempty(disconnected_poses(graph))
    @test_throws DisconnectedGraphError solve(graph)
    @test_throws DisconnectedGraphError compare(graph)
end
