"""Measurement graph, cost, residual, and Jacobian."""

"""
One relative-pose measurement Z ≈ X_i⁻¹ X_j.

`information` is the diagonal of Ω_ij. The edge contributes
rᵀ Ω r = || sqrt(Ω) Log(Z⁻¹ X_i⁻¹ X_j) ||².
`kind` is `:odom` or `:loop`. Dead reckoning uses only `:odom`.
"""
struct Edge
    i::Int
    j::Int
    Z::Pose
    information::NTuple{3,Float64}
    kind::Symbol
end

struct PoseGraph
    n::Int
    edges::Vector{Edge}
    truth::Union{Nothing,Vector{Pose}}
    function PoseGraph(n::Int, edges::Vector{Edge}, truth::Union{Nothing,Vector{Pose}} = nothing)
        n >= 1 || error("a pose graph needs at least one pose")
        for e in edges
            (1 <= e.i <= n && 1 <= e.j <= n && e.i != e.j) ||
                error("edge ($(e.i), $(e.j)) is outside a graph of $n poses")
            all(w -> w > 0, e.information) || error("information weights must be positive")
            (e.kind == :odom || e.kind == :loop) || error("edge kind must be :odom or :loop")
        end
        if truth !== nothing && length(truth) != n
            error("ground truth has $(length(truth)) poses, graph has $n")
        end
        return new(n, edges, truth)
    end
end

"""Poses that cannot be reached from pose 1. Empty when the gauge is enough."""
function disconnected_poses(graph::PoseGraph)
    parent = collect(1:graph.n)
    function find(a)
        while parent[a] != a
            parent[a] = parent[parent[a]]
            a = parent[a]
        end
        return a
    end
    for e in graph.edges
        parent[find(e.i)] = find(e.j)
    end
    root = find(1)
    return [i for i in 1:graph.n if find(i) != root]
end

struct DisconnectedGraphError <: Exception
    missing::Vector{Int}
end

function Base.showerror(io::IO, e::DisconnectedGraphError)
    print(io, "disconnected graph: poses ", e.missing,
          " are not connected to pose 1. The gauge-fixed problem has no solution, so no estimate is returned.")
end

function require_connected(graph::PoseGraph)
    missing = disconnected_poses(graph)
    isempty(missing) || throw(DisconnectedGraphError(missing))
    return nothing
end

"""Lie-algebra residual Log(Z⁻¹ X_i⁻¹ X_j), before weighting."""
function error_algebra(edge::Edge, poses::AbstractVector{Pose})
    predicted = between(poses[edge.i], poses[edge.j])
    return log_se2(between(edge.Z, predicted))
end

function whiten(information::NTuple{3,Float64}, φ::AbstractVector)
    return [sqrt(information[1]) * φ[1], sqrt(information[2]) * φ[2], sqrt(information[3]) * φ[3]]
end

"""Stacked whitened residuals, one 3-vector per edge."""
function residuals(graph::PoseGraph, poses::AbstractVector{Pose})
    r = zeros(3 * length(graph.edges))
    for (k, edge) in enumerate(graph.edges)
        r[3k - 2:3k] = whiten(edge.information, error_algebra(edge, poses))
    end
    return r
end

"""Sum of weighted squared residuals."""
function pose_cost(graph::PoseGraph, poses::AbstractVector{Pose})
    total = 0.0
    for edge in graph.edges
        w = whiten(edge.information, error_algebra(edge, poses))
        total += w[1]^2 + w[2]^2 + w[3]^2
    end
    return total
end

"""
Whitened residual vector and its Jacobian with respect to right perturbations
of poses 2…n. Pose 1 is not a variable.

For an edge i → j with φ = Log(Z⁻¹ X_i⁻¹ X_j) and T = X_i⁻¹ X_j,

    ∂φ/∂δ_j =  Jᵣ⁻¹(φ)
    ∂φ/∂δ_i = −Jᵣ⁻¹(φ) Ad(T⁻¹)

Rows are then scaled by sqrt(Ω), the same weights applied to φ.
"""
function residual_and_jacobian(graph::PoseGraph, poses::AbstractVector{Pose})
    m = length(graph.edges)
    nfree = graph.n - 1
    r = zeros(3m)
    J = zeros(3m, 3nfree)
    for (k, edge) in enumerate(graph.edges)
        rows = (3k - 2):(3k)
        predicted = between(poses[edge.i], poses[edge.j])
        φ = log_se2(between(edge.Z, predicted))
        r[rows] = whiten(edge.information, φ)
        Jrinv = right_jacobian_inverse(φ)
        scale = Diagonal(sqrt.(collect(edge.information)))
        if edge.j != 1
            cols = _free_cols(edge.j)
            J[rows, cols] .+= scale * Jrinv
        end
        if edge.i != 1
            cols = _free_cols(edge.i)
            J[rows, cols] .+= scale * (-Jrinv * Ad(inverse(predicted)))
        end
    end
    return r, J
end

"Columns of the state vector that belong to pose `p` (p ≥ 2)."
function _free_cols(p::Int)
    start = 3 * (p - 2) + 1
    return start:(start + 2)
end

"""
Central-difference Jacobian of the whitened residual, using the same right
perturbation X ← X ∘ Exp(ε e_k) as the optimizer.
"""
function jacobian_finite_difference(graph::PoseGraph, poses::AbstractVector{Pose}; ε::Float64 = 1e-6)
    r_plus = residuals(graph, poses) # placeholder, overwritten
    nfree = graph.n - 1
    J = zeros(length(r_plus), 3nfree)
    for p in 2:graph.n
        for d in 1:3
            step = zeros(3)
            step[d] = ε
            poses_plus = copy(poses)
            poses_minus = copy(poses)
            poses_plus[p] = compose(poses[p], exp_se2(step))
            step[d] = -ε
            poses_minus[p] = compose(poses[p], exp_se2(step))
            rp = residuals(graph, poses_plus)
            rm = residuals(graph, poses_minus)
            J[:, _free_cols(p)[d]] = (rp - rm) ./ (2ε)
        end
    end
    return J
end

"""Largest absolute entrywise gap between the analytic and finite-difference Jacobians."""
function jacobian_discrepancy(graph::PoseGraph, poses::AbstractVector{Pose}; ε::Float64 = 1e-6)
    _, J = residual_and_jacobian(graph, poses)
    Jn = jacobian_finite_difference(graph, poses; ε = ε)
    return maximum(abs.(J - Jn))
end

struct PoseError
    pos_rmse::Float64
    heading_rmse::Float64
    pos_max::Float64
    heading_max::Float64
end

"""
Position and heading error after both trajectories are left-multiplied so that
pose 1 is the identity. A global shift or rotation does not change the cost,
so it is not counted as error.
"""
function pose_error(estimate::AbstractVector{Pose}, truth::AbstractVector{Pose})
    length(estimate) == length(truth) || error("estimate and ground truth have different lengths")
    estimate = align_gauge(estimate)
    truth = align_gauge(truth)
    n = length(estimate)
    pos2 = 0.0
    heading2 = 0.0
    pos_max = 0.0
    heading_max = 0.0
    for i in 1:n
        dp = hypot(estimate[i].x - truth[i].x, estimate[i].y - truth[i].y)
        dθ = abs(wrap_angle(estimate[i].θ - truth[i].θ))
        pos2 += dp^2
        heading2 += dθ^2
        pos_max = max(pos_max, dp)
        heading_max = max(heading_max, dθ)
    end
    return PoseError(sqrt(pos2 / n), sqrt(heading2 / n), pos_max, heading_max)
end
