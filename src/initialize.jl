"""Dead-reckoning and spectral initial guesses."""

"""
Chain odometry edges outward from pose 1. Loop closures are ignored.

X₁ = I, and X_j = X_i ∘ Z_ij along each odometry edge i → j.
Poses the odometry chain never reaches stay at the identity.
"""
function dead_reckoning(graph::PoseGraph)
    children = [Tuple{Int,Pose}[] for _ in 1:graph.n]
    for edge in graph.edges
        edge.kind == :odom || continue
        push!(children[edge.i], (edge.j, edge.Z))
    end
    poses = [IDENTITY for _ in 1:graph.n]
    seen = falses(graph.n)
    seen[1] = true
    queue = Int[1]
    head = 1
    while head <= length(queue)
        i = queue[head]
        head += 1
        for (j, Z) in children[i]
            seen[j] && continue
            poses[j] = compose(poses[i], Z)
            seen[j] = true
            push!(queue, j)
        end
    end
    return poses
end

"""Nearest rotation in SO(2) to a 2×2 matrix, in the Frobenius norm."""
function project_so2(A::AbstractMatrix)
    F = svd(A)
    U = Matrix(F.U)
    if det(U * F.V') < 0
        U[:, 2] *= -1
    end
    return U * F.V'
end

"""
Spectral initialization.

The rotational measurements define the connection Laplacian L ∈ R^{2n×2n}.
With Y stacking the unknown rotations, tr(Yᵀ L Y) equals Σ κ_ij ||R_i R_ij − R_j||²
(up to a constant). The two eigenvectors belonging to the smallest eigenvalues
are reshaped into n blocks of 2×2 and each block is projected onto SO(2).

Given those rotations, every translation measurement is linear,

    t_j − t_i = R_i t_ij,

so the positions are a linear least-squares problem with t₁ = 0. Each pair
(R_i, t_i) is then a pose in SE(2). Pose 1 is the identity.
"""
function spectral_initialization(graph::PoseGraph)
    n = graph.n
    L = zeros(2n, 2n)
    for edge in graph.edges
        κ = edge.information[3]
        R = rotation_matrix(edge.Z.θ)
        ii = (2 * (edge.i - 1) + 1):(2 * edge.i)
        jj = (2 * (edge.j - 1) + 1):(2 * edge.j)
        L[ii, ii] += κ * I(2)
        L[jj, jj] += κ * I(2)
        L[ii, jj] += -κ * R
        L[jj, ii] += -κ * R'
    end
    factors = eigen(Symmetric(L))
    V = Matrix(factors.vectors[:, 1:2])
    # One common right-multiplication by a reflection would put every block in
    # the wrong component of O(2). Flipping one basis vector prevents that.
    if det(@view V[1:2, 1:2]) < 0
        V[:, 2] .*= -1
    end
    rotations = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        block = @view V[(2i - 1):(2i), 1:2]
        # The kernel stores R_iᵀ Q, so transpose after the projection.
        rotations[i] = project_so2(block)'
    end
    gauge = rotations[1]'
    for i in 1:n
        rotations[i] = gauge * rotations[i]
    end
    translations = _translation_least_squares(graph, rotations)
    poses = Vector{Pose}(undef, n)
    for i in 1:n
        R = rotations[i]
        poses[i] = Pose(translations[i, 1], translations[i, 2], atan(R[2, 1], R[1, 1]))
    end
    poses[1] = IDENTITY
    return poses
end

"""Weighted linear least squares for t_j − t_i = R_i t_ij, with t₁ fixed at 0."""
function _translation_least_squares(graph::PoseGraph, rotations::Vector{Matrix{Float64}})
    n = graph.n
    nedges = length(graph.edges)
    A = zeros(2nedges, 2 * (n - 1))
    b = zeros(2nedges)
    for (k, edge) in enumerate(graph.edges)
        rows = (2k - 1):(2k)
        # sqrt(Ω) on the measurement, assuming σ_x and σ_y are the same scale.
        w = sqrt(min(edge.information[1], edge.information[2]))
        rhs = rotations[edge.i] * [edge.Z.x, edge.Z.y]
        if edge.i != 1
            cols = (2 * (edge.i - 2) + 1):(2 * (edge.i - 2) + 2)
            A[rows, cols] = [-w 0.0; 0.0 -w]
        end
        if edge.j != 1
            cols = (2 * (edge.j - 2) + 1):(2 * (edge.j - 2) + 2)
            A[rows, cols] += [w 0.0; 0.0 w]
        end
        b[rows] = w .* rhs
    end
    solution = A \ b
    translations = zeros(n, 2)
    for i in 2:n
        translations[i, :] = solution[(2 * (i - 2) + 1):(2 * (i - 2) + 2)]
    end
    return translations
end
