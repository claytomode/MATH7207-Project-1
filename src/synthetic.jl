"""Synthetic pose graphs. Measurements are the true relative pose plus Gaussian noise."""

function _measurement(truth::Vector{Pose}, i::Int, j::Int, σ::NTuple{3,Float64}, kind::Symbol, rng::AbstractRNG)
    Z = between(truth[i], truth[j])
    if any(s -> s > 0, σ)
        ε = [σ[1] * randn(rng), σ[2] * randn(rng), σ[3] * randn(rng)]
        Z = compose(Z, exp_se2(ε))
    end
    information = ntuple(k -> σ[k] == 0 ? 1.0 : 1 / σ[k]^2, 3)
    return Edge(i, j, Z, information, kind)
end

"""
Poses of a robot driving once around a circle, then left-multiplied so pose 1
is the identity. Heading is tangent to the circle.
"""
function circle_poses(n::Int, radius::Float64)
    n >= 3 || error("a loop needs at least 3 poses")
    raw = Vector{Pose}(undef, n)
    for k in 0:(n - 1)
        ψ = 2π * k / n
        raw[k + 1] = Pose(radius * cos(ψ), radius * sin(ψ), wrap_angle(ψ + π / 2))
    end
    return align_gauge(raw)
end

"""
A single loop. Odometry is the chain 1→2→…→n. One loop closure is n→1.
Noise is Gaussian in the Lie algebra of each relative pose. `σ = 0` is exact.
"""
function loop_graph(;
    n::Int = 20,
    radius::Float64 = 10.0,
    σ_odom::NTuple{3,Float64} = (0.0, 0.0, 0.0),
    σ_loop::NTuple{3,Float64} = (0.0, 0.0, 0.0),
    seed::Int = 1,
)
    truth = circle_poses(n, radius)
    rng = MersenneTwister(seed)
    edges = Edge[]
    for i in 1:(n - 1)
        push!(edges, _measurement(truth, i, i + 1, σ_odom, :odom, rng))
    end
    push!(edges, _measurement(truth, n, 1, σ_loop, :loop, rng))
    return PoseGraph(n, edges, truth)
end

"""Exact loop. Both starts should recover the ground truth."""
noiseless_graph(; n::Int = 12, radius::Float64 = 8.0) = loop_graph(; n = n, radius = radius, seed = 1)

"""
Moderate odometry noise and one loop closure. Dead reckoning drifts, but the
drift stays inside the basin of the true loop, so both refinements finish near
the ground truth. The spectral start begins closer.
"""
function moderate_graph()
    return loop_graph(;
        n = 30,
        radius = 10.0,
        σ_odom = (0.05, 0.05, 0.04),
        σ_loop = (0.02, 0.02, 0.01),
        seed = 4,
    )
end

"""
Robot travels a grid in boustrophedon order. Consecutive poses are odometry.
Every other pair of grid neighbors is a loop closure.

`hard_graph` uses large heading noise on the odometry. On that instance,
Levenberg–Marquardt from dead reckoning stops at a local minimum, and the
spectral start recovers the grid. A single loop with moderate noise did not
split the two starts; the extra loop closures are what make the basin matter.
"""
function grid_graph(;
    rows::Int = 8,
    cols::Int = 8,
    spacing::Float64 = 2.0,
    σ_odom::NTuple{3,Float64} = (0.05, 0.05, 0.7),
    σ_loop::NTuple{3,Float64} = (0.03, 0.03, 0.02),
    seed::Int = 2,
)
    order = Tuple{Int,Int}[]
    for r in 1:rows
        columns = isodd(r) ? (1:cols) : (cols:-1:1)
        for c in columns
            push!(order, (r, c))
        end
    end
    n = length(order)
    raw = Vector{Pose}(undef, n)
    for (k, (r, c)) in enumerate(order)
        heading = isodd(r) ? 0.0 : π
        raw[k] = Pose((c - 1) * spacing, (r - 1) * spacing, heading)
    end
    truth = align_gauge(raw)
    rng = MersenneTwister(seed)
    edges = Edge[]
    for i in 1:(n - 1)
        push!(edges, _measurement(truth, i, i + 1, σ_odom, :odom, rng))
    end
    location = Dict(order[k] => k for k in 1:n)
    for (k, (r, c)) in enumerate(order)
        for (rr, cc) in ((r + 1, c), (r, c + 1))
            haskey(location, (rr, cc)) || continue
            j = location[(rr, cc)]
            abs(k - j) == 1 && continue
            i, jj = minmax(k, j)
            push!(edges, _measurement(truth, i, jj, σ_loop, :loop, rng))
        end
    end
    return PoseGraph(n, edges, truth)
end

hard_graph() = grid_graph()

"""
An Archimedean spiral. Odometry follows the curve. A loop closure joins a pose
to the pose one turn earlier, so the gray chords are the gaps between arms.

`hard_spiral` is the tuned instance: large odometry heading noise. Dead
reckoning unwinds the spiral, and the spectral start is what puts
Levenberg–Marquardt back on the curve.
"""
function spiral_graph(;
    turns::Int = 4,
    per_turn::Int = 20,
    spacing::Float64 = 1.6,
    loop_every::Int = 1,
    σ_odom::NTuple{3,Float64} = (0.04, 0.04, 0.45),
    σ_loop::NTuple{3,Float64} = (0.03, 0.03, 0.02),
    seed::Int = 1,
)
    turns >= 2 && per_turn >= 8 || error("a spiral needs at least two turns")
    loop_every >= 1 || error("loop_every must be positive")
    n = turns * per_turn
    raw = Vector{Pose}(undef, n)
    for k in 0:(n - 1)
        θ = 2π * k / per_turn
        radius = spacing * (0.6 + k / per_turn)
        raw[k + 1] = Pose(radius * cos(θ), radius * sin(θ), 0.0)
    end
    for k in 1:(n - 1)
        dx = raw[k + 1].x - raw[k].x
        dy = raw[k + 1].y - raw[k].y
        raw[k] = Pose(raw[k].x, raw[k].y, atan(dy, dx))
    end
    raw[n] = Pose(raw[n].x, raw[n].y, raw[n - 1].θ)
    truth = align_gauge(raw)
    rng = MersenneTwister(seed)
    edges = Edge[]
    for i in 1:(n - 1)
        push!(edges, _measurement(truth, i, i + 1, σ_odom, :odom, rng))
    end
    for j in (per_turn + 1):n
        (j - per_turn) % loop_every == 0 || continue
        push!(edges, _measurement(truth, j - per_turn, j, σ_loop, :loop, rng))
    end
    return PoseGraph(n, edges, truth)
end

hard_spiral() = spiral_graph(;
    turns = 4,
    per_turn = 24,
    spacing = 1.6,
    loop_every = 1,
    σ_odom = (0.04, 0.04, 0.50),
    σ_loop = (0.03, 0.03, 0.02),
    seed = 3,
)

"""
Three poses and three edges: the triangle used to check the residual and the
Jacobian against a finite difference. Measurements are exact. Evaluate the
Jacobian at a perturbed state, not at the root, so the residual is nonzero.
"""
function toy_graph()
    truth = [IDENTITY, Pose(1.2, 0.15, 0.35), Pose(0.4, 1.1, -0.55)]
    edges = Edge[
        Edge(1, 2, between(truth[1], truth[2]), (4.0, 4.0, 9.0), :odom),
        Edge(2, 3, between(truth[2], truth[3]), (4.0, 4.0, 9.0), :odom),
        Edge(1, 3, between(truth[1], truth[3]), (2.0, 2.0, 6.0), :loop),
    ]
    return PoseGraph(3, edges, truth)
end

"A state near the toy graph's ground truth, used for the finite-difference check."
function toy_evaluation_state(graph::PoseGraph)
    poses = copy(graph.truth)
    poses[2] = compose(poses[2], exp_se2([0.12, -0.07, 0.25]))
    poses[3] = compose(poses[3], exp_se2([-0.05, 0.09, -0.18]))
    return poses
end

"""Two separate odometry chains. Pose 1's component does not contain poses 4 and 5."""
function disconnected_graph()
    truth = [IDENTITY, Pose(1.0, 0.0, 0.1), Pose(2.0, 0.2, 0.2), Pose(10.0, 0.0, 0.0), Pose(11.0, 0.3, -0.2)]
    edges = Edge[
        Edge(1, 2, between(truth[1], truth[2]), (1.0, 1.0, 1.0), :odom),
        Edge(2, 3, between(truth[2], truth[3]), (1.0, 1.0, 1.0), :odom),
        Edge(4, 5, between(truth[4], truth[5]), (1.0, 1.0, 1.0), :odom),
    ]
    return PoseGraph(5, edges, truth)
end
