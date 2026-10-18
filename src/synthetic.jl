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
A maze built by a randomized depth-first search, with a few extra passages
knocked through so the corridors are not a pure tree. The robot walks every
corridor and backtracks, so the trajectory traces the maze. Odometry follows
that walk. A loop closure is added when a later pose can see back through an
open passage to a much earlier visit.

`crazy_maze` is the tuned instance: large odometry heading noise, and the
spectral start is what puts Levenberg–Marquardt back in the corridors.
Returns the pose graph and the wall segments in the same gauge as the truth,
each wall as `(x1, y1, x2, y2)`.
"""
function maze_graph(;
    rows::Int = 10,
    cols::Int = 10,
    spacing::Float64 = 2.0,
    extra_loops::Float64 = 0.18,
    revisit_gap::Int = 24,
    σ_odom::NTuple{3,Float64} = (0.04, 0.04, 0.55),
    σ_loop::NTuple{3,Float64} = (0.03, 0.03, 0.015),
    seed::Int = 1,
)
    rows >= 3 && cols >= 3 || error("a maze needs at least a 3 by 3 block of cells")
    rng = MersenneTwister(seed)
    passages = _maze_passages(rows, cols, rng, extra_loops)
    walk = _maze_walk(rows, cols, passages)
    raw = Vector{Pose}(undef, length(walk))
    for (k, (r, c)) in enumerate(walk)
        x = (c - 0.5) * spacing
        y = (r - 0.5) * spacing
        if k == 1
            θ = 0.0
        else
            pr, pc = walk[k - 1]
            θ = atan(Float64(r - pr), Float64(c - pc))
        end
        raw[k] = Pose(x, y, θ)
    end
    gauge = inverse(raw[1])
    truth = align_gauge(raw)
    edges = Edge[]
    for i in 1:(length(walk) - 1)
        push!(edges, _measurement(truth, i, i + 1, σ_odom, :odom, rng))
    end
    visits = Dict{NTuple{2,Int},Vector{Int}}()
    for (k, cell) in enumerate(walk)
        push!(get!(visits, cell, Int[]), k)
    end
    neighbors = _passage_neighbors(rows, cols, passages)
    seen_pairs = Set{Tuple{Int,Int}}()
    for j in 1:length(walk)
        cell = walk[j]
        for other in neighbors[cell]
            earlier = get(visits, other, Int[])
            idx = findlast(k -> k <= j - revisit_gap, earlier)
            idx === nothing && continue
            i = earlier[idx]
            pair = (i, j)
            pair in seen_pairs && continue
            push!(seen_pairs, pair)
            push!(edges, _measurement(truth, i, j, σ_loop, :loop, rng))
        end
    end
    walls = NTuple{4,Float64}[]
    for (x1, y1, x2, y2) in _maze_walls(rows, cols, spacing, passages)
        ax, ay = _map_point(gauge, x1, y1)
        bx, by = _map_point(gauge, x2, y2)
        push!(walls, (ax, ay, bx, by))
    end
    return PoseGraph(length(walk), edges, truth), walls
end

function crazy_maze()
    return maze_graph(;
        rows = 10,
        cols = 10,
        spacing = 2.0,
        extra_loops = 0.18,
        revisit_gap = 16,
        σ_odom = (0.04, 0.04, 0.45),
        σ_loop = (0.03, 0.03, 0.015),
        seed = 2,
    )
end

function _map_point(pose::Pose, x::Float64, y::Float64)
    c = cos(pose.θ)
    s = sin(pose.θ)
    return (pose.x + c * x - s * y, pose.y + s * x + c * y)
end

function _cell_key(a::NTuple{2,Int}, b::NTuple{2,Int})
    return a <= b ? (a, b) : (b, a)
end

function _maze_passages(rows::Int, cols::Int, rng::AbstractRNG, extra_loops::Float64)
    visited = falses(rows, cols)
    stack = NTuple{2,Int}[(1, 1)]
    visited[1, 1] = true
    passages = Set{Tuple{NTuple{2,Int},NTuple{2,Int}}}()
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while !isempty(stack)
        r, c = stack[end]
        choices = NTuple{2,Int}[]
        for (dr, dc) in directions
            rr, cc = r + dr, c + dc
            if 1 <= rr <= rows && 1 <= cc <= cols && !visited[rr, cc]
                push!(choices, (rr, cc))
            end
        end
        if isempty(choices)
            pop!(stack)
            continue
        end
        nxt = choices[rand(rng, 1:length(choices))]
        visited[nxt[1], nxt[2]] = true
        push!(passages, _cell_key((r, c), nxt))
        push!(stack, nxt)
    end
    blocked = Tuple{NTuple{2,Int},NTuple{2,Int}}[]
    for r in 1:rows, c in 1:cols
        for nxt in ((r + 1, c), (r, c + 1))
            (1 <= nxt[1] <= rows && 1 <= nxt[2] <= cols) || continue
            key = _cell_key((r, c), nxt)
            key in passages || push!(blocked, key)
        end
    end
    shuffle!(rng, blocked)
    n_open = clamp(round(Int, extra_loops * length(blocked)), 0, length(blocked))
    for key in blocked[1:n_open]
        push!(passages, key)
    end
    return passages
end

function _passage_neighbors(rows::Int, cols::Int, passages)
    neighbors = Dict{NTuple{2,Int},Vector{NTuple{2,Int}}}()
    for r in 1:rows, c in 1:cols
        neighbors[(r, c)] = NTuple{2,Int}[(r, c)]
    end
    for (a, b) in passages
        push!(neighbors[a], b)
        push!(neighbors[b], a)
    end
    return neighbors
end

function _maze_walk(rows::Int, cols::Int, passages)
    adj = Dict{NTuple{2,Int},Vector{NTuple{2,Int}}}()
    for r in 1:rows, c in 1:cols
        adj[(r, c)] = NTuple{2,Int}[]
    end
    for (a, b) in passages
        push!(adj[a], b)
        push!(adj[b], a)
    end
    for cell in keys(adj)
        sort!(adj[cell])
    end
    walk = NTuple{2,Int}[(1, 1)]
    seen = Set{NTuple{2,Int}}([(1, 1)])
    function dfs(cell)
        for nxt in adj[cell]
            nxt in seen && continue
            push!(seen, nxt)
            push!(walk, nxt)
            dfs(nxt)
            push!(walk, cell)
        end
        return nothing
    end
    dfs((1, 1))
    return walk
end

function _maze_walls(rows::Int, cols::Int, spacing::Float64, passages)
    open_wall = Set(passages)
    segs = NTuple{4,Float64}[]
    for r in 1:rows, c in 1:cols
        x0 = (c - 1) * spacing
        y0 = (r - 1) * spacing
        x1 = c * spacing
        y1 = r * spacing
        if r == 1
            push!(segs, (x0, y0, x1, y0))
        end
        if c == 1
            push!(segs, (x0, y0, x0, y1))
        end
        north = r == rows || !(_cell_key((r, c), (r + 1, c)) in open_wall)
        east = c == cols || !(_cell_key((r, c), (r, c + 1)) in open_wall)
        north && push!(segs, (x0, y1, x1, y1))
        east && push!(segs, (x1, y0, x1, y1))
    end
    return segs
end

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
