"""Levenberg–Marquardt on SE(2) poses."""

struct SolveResult
    init::Symbol
    init_poses::Vector{Pose}
    poses::Vector{Pose}
    init_cost::Float64
    cost::Float64
    iterations::Int
    cost_history::Vector{Float64}
    converged::Bool
    gradient_norm::Float64
end

struct Comparison
    dead_reckoning::SolveResult
    spectral::SolveResult
end

"""X_p ← X_p ∘ Exp(δ_p) for every free pose. Pose 1 stays the identity."""
function retract(poses::AbstractVector{Pose}, δ::AbstractVector)
    length(δ) == 3 * (length(poses) - 1) ||
        error("update has length $(length(δ)), expected $(3 * (length(poses) - 1))")
    out = copy(poses)
    out[1] = IDENTITY
    for p in 2:length(poses)
        cols = _free_cols(p)
        out[p] = compose(poses[p], exp_se2(@view δ[cols]))
    end
    return out
end

"""
Levenberg–Marquardt from a starting guess.

Each iteration builds r and J, solves (JᵀJ + λ I) δ = −Jᵀr, and tries
X ← X ∘ Exp(δ). A cheaper step is accepted and λ is divided by `λ_factor`.
A step that does not lower the cost is rejected and λ is multiplied by
`λ_factor`. Pose 1 is held at the identity. Stops when ||Jᵀr|| < `grad_tol`
or `max_iters` is reached.

`cost_history` starts with the cost of the initial guess. Rejected steps
leave the cost unchanged, so a flat stretch in the history is a rejected step.
"""
function levenberg_marquardt(
    graph::PoseGraph,
    poses0::AbstractVector{Pose};
    max_iters::Int = 40,
    grad_tol::Float64 = 1e-8,
    λ0::Float64 = 1e-3,
    λ_factor::Float64 = 10.0,
    λ_max::Float64 = 1e12,
)
    length(poses0) == graph.n || error("starting guess has $(length(poses0)) poses, graph has $(graph.n)")
    poses = align_gauge(poses0)
    start = copy(poses)
    history = Float64[pose_cost(graph, poses)]
    λ = λ0
    iterations = 0
    converged = false
    gradient_norm = Inf
    for _ in 1:max_iters
        r, J = residual_and_jacobian(graph, poses)
        g = J' * r
        gradient_norm = norm(g)
        if gradient_norm < grad_tol
            converged = true
            break
        end
        δ = (J' * J + λ * I) \ (-g)
        candidate = retract(poses, δ)
        new_cost = pose_cost(graph, candidate)
        iterations += 1
        if new_cost < history[end]
            poses = candidate
            push!(history, new_cost)
            λ = max(λ / λ_factor, 1e-16)
        else
            push!(history, history[end])
            λ *= λ_factor
            if λ > λ_max
                break
            end
        end
    end
    r_final, J_final = residual_and_jacobian(graph, poses)
    gradient_norm = norm(J_final' * r_final)
    converged = converged || gradient_norm < grad_tol
    return SolveResult(
        :given,
        start,
        poses,
        history[1],
        history[end],
        iterations,
        history,
        converged,
        gradient_norm,
    )
end

"""
Solve the pose graph from `:dead_reckoning` or `:spectral`.

Throws `DisconnectedGraphError` when some pose is not in the component of pose 1.
"""
function solve(graph::PoseGraph; init::Symbol = :dead_reckoning, kwargs...)
    require_connected(graph)
    if init == :dead_reckoning
        poses0 = dead_reckoning(graph)
    elseif init == :spectral
        poses0 = spectral_initialization(graph)
    else
        error("unknown initialization :$init")
    end
    result = levenberg_marquardt(graph, poses0; kwargs...)
    return SolveResult(
        init,
        result.init_poses,
        result.poses,
        result.init_cost,
        result.cost,
        result.iterations,
        result.cost_history,
        result.converged,
        result.gradient_norm,
    )
end

"Run Levenberg–Marquardt from both starting guesses."
function compare(graph::PoseGraph; kwargs...)
    return Comparison(
        solve(graph; init = :dead_reckoning, kwargs...),
        solve(graph; init = :spectral, kwargs...),
    )
end

init_label(init::Symbol) = init == :dead_reckoning ? "dead reckoning" : "spectral"
