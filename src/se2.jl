"""Rigid motions of the plane."""

struct Pose
    x::Float64
    y::Float64
    θ::Float64
end

const IDENTITY = Pose(0.0, 0.0, 0.0)

"""Wrap an angle in radians into (−π, π]."""
function wrap_angle(θ::Real)
    return mod(Float64(θ) + π, 2π) - π
end

"""X ∘ Y. Apply Y first, then X."""
function compose(a::Pose, b::Pose)
    c = cos(a.θ)
    s = sin(a.θ)
    return Pose(
        a.x + c * b.x - s * b.y,
        a.y + s * b.x + c * b.y,
        wrap_angle(a.θ + b.θ),
    )
end

function inverse(a::Pose)
    c = cos(a.θ)
    s = sin(a.θ)
    return Pose(-c * a.x - s * a.y, s * a.x - c * a.y, wrap_angle(-a.θ))
end

"""Pose of `b` expressed in the frame of `a`, equal to a⁻¹ ∘ b."""
between(a::Pose, b::Pose) = compose(inverse(a), b)

"""
Exponential map se(2) → SE(2).

`ξ = (ρx, ρy, θ)` is a Lie algebra coordinate. The translation of the resulting
pose is V(θ) ρ, with V(θ) = [[a, −b], [b, a]], a = sinθ/θ, b = (1 − cosθ)/θ.
"""
function exp_se2(ξ::AbstractVector)
    ρx = Float64(ξ[1])
    ρy = Float64(ξ[2])
    θ = Float64(ξ[3])
    if abs(θ) < 1e-8
        a = 1.0 - θ^2 / 6
        b = θ / 2 - θ^3 / 24
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ
    end
    return Pose(a * ρx - b * ρy, b * ρx + a * ρy, wrap_angle(θ))
end

"""
Logarithm SE(2) → se(2). Inverse of `exp_se2` for headings in (−π, π].

V(θ)⁻¹ = [[α, β], [−β, α]] with α = (θ/2) cot(θ/2) and β = θ/2.
"""
function log_se2(p::Pose)
    θ = wrap_angle(p.θ)
    if abs(θ) < 1e-8
        α = 1.0 - θ^2 / 12
        β = θ / 2
    else
        half = θ / 2
        α = half * cos(half) / sin(half)
        β = half
    end
    return [α * p.x + β * p.y, -β * p.x + α * p.y, θ]
end

"""
Adjoint Ad_X : se(2) → se(2), the matrix of X Exp(ξ) X⁻¹ = Exp(Ad_X ξ).

    | cosθ  −sinθ   y |
    | sinθ   cosθ  −x |
    |  0      0     1 |
"""
function Ad(p::Pose)
    c = cos(p.θ)
    s = sin(p.θ)
    return [c -s p.y; s c -p.x; 0.0 0.0 1.0]
end

"""
Inverse right Jacobian of SE(2).

For φ = Log(E), d/dt |₀ Log(E ∘ Exp(tη)) = Jᵣ⁻¹(φ) η.
Closed form from the Bernoulli series of ad_φ, which truncates because
ad_φ³ = −θ² ad_φ.
"""
function right_jacobian_inverse(ξ::AbstractVector)
    x = Float64(ξ[1])
    y = Float64(ξ[2])
    θ = Float64(ξ[3])
    if abs(θ) < 1e-6
        return [
            1 - θ^2 / 12    -θ / 2         y / 2 + θ * x / 12
            θ / 2         1 - θ^2 / 12    -x / 2 + θ * y / 12
            0.0            0.0             1.0
        ]
    end
    s = sin(θ)
    c = cos(θ)
    α = (θ / 2) * cos(θ / 2) / sin(θ / 2)
    β = θ / 2
    A = (1 - c) / θ^2
    Bθ = (θ - s) / θ^2
    vx = -A * y + Bθ * x
    vy = A * x + Bθ * y
    # Third column is −C⁻¹ v, with C⁻¹ = [[α, −β], [β, α]].
    wx = -(α * vx - β * vy)
    wy = -(β * vx + α * vy)
    return [α -β wx; β α wy; 0.0 0.0 1.0]
end

"""Left-multiply every pose by X₁⁻¹, then pin pose 1 at the identity."""
function align_gauge(poses::AbstractVector{Pose})
    g = inverse(poses[1])
    out = Pose[compose(g, p) for p in poses]
    out[1] = IDENTITY
    return out
end

function rotation_matrix(θ::Real)
    c = cos(θ)
    s = sin(θ)
    return [c -s; s c]
end
