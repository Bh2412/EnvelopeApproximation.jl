using Bessels
import Bessels: besselj!
import EnvelopeApproximation.ChebyshevCFT: values!, chebyshev_coeffs!, fourier_mode, scale, translation, multiplication_weights, chebyshevpoints, First3MomentsChebyshevPlan, inverse_chebyshev_weight, inverse_u
using FastTransforms
using BenchmarkTools
using LinearAlgebra
using Test
using StaticArrays
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.GeometricStressEnergyTensor: ∫_ϕ, upper_right, _buffers, PeriodicInterval, IntersectionDome
import EnvelopeApproximation.BubblesEvolution: BallSpace, BubblesSnapShot
using QuadGK
using HCubature

struct VectorChebyshevPlan{N, K}
    points:: Vector{Float64}
    coeffs_buffer:: Matrix{Float64}
    bessels_buffer:: Vector{Float64}
    multiplication_weights:: Vector{ComplexF64}
    multiplication_buffer:: Vector{ComplexF64}
    transform_plan!:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}
    mode_buffer:: Vector{ComplexF64}

    function VectorChebyshevPlan{N, K}() where {N, K}
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Matrix{Float64}(undef, N, K)
        bessels_buffer = Vector{Float64}(undef, N)
        weights = multiplication_weights(N)
        multiplication_buffer = Vector{ComplexF64}(undef, N)
        transform_plan! = plan_chebyshevtransform!(zeros(N), Val(1))
        mode_buffer = Vector{ComplexF64}(undef, K)
        return new{N, K}(points, coeffs_buffer,  
                         bessels_buffer, weights, multiplication_buffer, transform_plan!, mode_buffer)
    end
end

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: VectorChebyshevPlan{N, K}) where {N, K}
    scale_factor = scale(a, b)
    t = translation(a, b)
    @inbounds for (i, u) in enumerate(chebyshev_plan.points)
        icw = inverse_chebyshev_weight(u) 
        @views @. chebyshev_plan.coeffs_buffer[i, :] = $f($inverse_u(u, scale_factor, t)) * icw
    end     
end

export chebyshev_coeffs!

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, 
                           chebyshev_plan:: VectorChebyshevPlan{N, K}) where {N, K}
    values!(f, a, b, chebyshev_plan)
    @inbounds for i in 1:K
        chebyshev_plan.transform_plan! * (@views chebyshev_plan.coeffs_buffer[:, i])
    end 
end

export fourier_mode

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: VectorChebyshevPlan{N, K}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: Vector{ComplexF64} where {N, K}
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N-1), k̃)
    e = cis(-k * translation) * scale
    @. chebyshev_plan.multiplication_buffer = e * chebyshev_plan.bessels_buffer * chebyshev_plan.multiplication_weights
    @inbounds for i in 1:K
        chebyshev_plan.mode_buffer[i] = (@views chebyshev_plan.coeffs_buffer[:, i]) ⋅ chebyshev_plan.multiplication_buffer
    end
    return chebyshev_plan.mode_buffer
end

function fourier_modes(ks:: AbstractVector{Float64}, f, a:: Real, b:: Real, 
    plan:: VectorChebyshevPlan{N, K}):: Matrix{ComplexF64} where {N, K}
    M = Matrix{ComplexF64}(undef, length(ks), K)
    chebyshev_coeffs!(f, a, b, plan)
    s = scale(a, b)
    t = translation(a, b)
    for (i, k) in enumerate(ks)
        @views M[i, :] .= fourier_mode(k, plan, s, t)
    end
    return M
end

function fourier_modes(ks:: AbstractVector{Float64}, f, a:: Real, b:: Real, 
    plan:: First3MomentsChebyshevPlan{N}):: Matrix{ComplexF64} where N
    M = Matrix{ComplexF64}(undef, length(ks), 3)
    chebyshev_coeffs!(f, a, b, plan)
    s = scale(a, b)
    t = translation(a, b)
    for (i, k) in enumerate(ks)
        @views M[i, :] .= fourier_mode(k, plan, s, t)
    end
    return M
end


begin

vector_chebyshev_plan = VectorChebyshevPlan{32, 3}()
moments_chebyshev_plan = First3MomentsChebyshevPlan{32}()

f(x) = exp(-x^2)
vector_f(x) = begin
    e = exp(-x^2)
    return [e, e * x, e * x ^ 2]
end

ks = range(0., 10., 1_000)

vector_modes = fourier_modes(ks, vector_f, -1., 1., vector_chebyshev_plan)
moments_modes = fourier_modes(ks, f, -1., 1., moments_chebyshev_plan)

@test max(((vector_modes .- moments_modes) .|> abs)...) < 1e-13

end

begin

vector_chebyshev_plan = VectorChebyshevPlan{32, 3}()
moments_chebyshev_plan = First3MomentsChebyshevPlan{32}()

f(x) = exp(-x^2)
vector_f(x) = begin
    e = exp(-x^2)
    return [e, e * x, e * x ^ 2]
end

a, b = 0.5, 3.
ks = range(0., 10., 1_000)

vector_modes = fourier_modes(ks, vector_f, a, b, vector_chebyshev_plan)
moments_modes = fourier_modes(ks, f, a, b, moments_chebyshev_plan)

@test max(((vector_modes .- moments_modes) .|> abs)...) < 1e-13
    
end

# @btime fourier_modes($ks, $vector_f, -1., 1., $vector_chebyshev_plan)
# @btime fourier_modes($ks, $f, -1., 1., $moments_chebyshev_plan)

# @profview for _ in 1:10_000 fourier_modes(ks, vector_f, -1., 1., vector_chebyshev_plan) end
# @profview for _ in 1:10_000 fourier_modes(ks, f, -1., 1., moments_chebyshev_plan) end

struct x̂_ix̂_j
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

x̂_ix̂_j(n:: Int64) = x̂_ix̂_j(_buffers(n)...)

function (f:: x̂_ix̂_j)(μ:: Float64, bubble:: Bubble, 
                      intersection_domes:: Vector{IntersectionDome}):: MVector{Float64}
    V = zeros(MVector{6, Float64})
    periodic_intervals = ring_domes_intersection!(μ, bubble.radius, intersection_domes, 
                                                  f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
    @inbounds for interval in periodic_intervals
        V .+= ∫_ϕ(upper_right, μ, interval.ϕ1, interval.ϕ1 + interval.Δ)
    end
    return V
end

function bubble_Tij_contribution!(V:: AbstractMatrix{ComplexF64},
                                  ks:: AbstractVector{Float64}, 
                                  bubble:: Bubble, 
                                  domes:: Vector{IntersectionDome}, 
                                  chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
                                  _x̂_ix̂_j:: x̂_ix̂_j; 
                                  ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert size(V) == (length(ks), 6) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    @inbounds for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _x̂_ix̂_j(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        @inbounds for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3)
            @. V[i, :] += e * fourier_mode(k, chebyshev_plan, s, t) # ∂_iφ∂_jφ contribution
        end
    end
    return V
end

function Tij(ks:: AbstractVector{Float64}, 
             bubbles:: AbstractVector{Bubble}, 
             ball_space:: BallSpace,
             chebyshev_plan:: VectorChebyshevPlan{N, 6},
             _x̂_ix̂_j:: x̂_ix̂_j;
             ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    V = zeros(ComplexF64, length(ks), 6)
    domes = intersection_domes(bubbles, ball_space)
    @inbounds for (bubble_index, _domes) in domes
        bubble_Tij_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                 chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
    end
    return V
end

const symmetric_tensor_indices:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2=> (1, 2), 3=> (1, 3), 4 =>(2, 2), 5 =>(2, 3), 6 => (3, 3))
const inverse_symmetric_tensor_indices:: Dict{Tuple{Int, Int}, Int} = Dict(zip(values(symmetric_tensor_indices), keys(symmetric_tensor_indices)))

function symmetric_dot(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
    r = 0.
    for ĩ in 1:6
        (i, j) = symmetric_tensor_indices[ĩ]
        (i == j) && (r += (T1[ĩ])' * T2[ĩ]); continue
        r += 2 * (T1[ĩ])' * T2[ĩ]
    end
    return r
end

function δ(T:: AbstractVector{ComplexF64}):: ComplexF64
    return T[1] + T[4] + T[6]
end

function zz(T:: AbstractVector{ComplexF64})
    return T[6]
end

function Λ(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
    r = 0.
    r += symmetric_dot(T1, T2)
    r += (-2) * @views (T1[[3, 5, 6]] ⋅ T2[[3, 5, 6]])
    zz1 = zz(T1)'
    zz2 = zz(T2)
    δ1 = δ(T1)'
    δ2 = δ(T2)
    r += (1. / 2) * zz1 * zz2
    r += (-1. / 2) * δ1 * δ2
    r += (1. / 2) * δ1 * zz2
    r += (1. / 2) * zz1 * δ2
    return r
end

function Λ(T:: AbstractVector{ComplexF64}):: ComplexF64
    return Λ(T, T)
end

function _TΛT(t:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
              ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
              _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1.):: Vector{ComplexF64} where N
    # Eq. 15 in Kosowsky and Turner
    T = Tij(ωs, current_bubble(snapshot, t), ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
    return @. Λ($eachrow(T)) * (2G * (ωs ^ 2))
end

function TΛT(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
             ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
             _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
    # Eq. 15 in Kosowsky and Turner
    function f(t:: Float64):: Vector{ComplexF64}
        return @. (cis(ωs * t) / 2π) * $_TΛT(t, ωs, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G)
    end
    return quadgk(f, 0., snapshot.t; kwargs...)
end

function integrand(ωs:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot,
                   ball_space:: BallSpace, 
                   chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                   _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., 
                   G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
    rot = align_ẑ(n̂(ΦΘ))
    θ = ΦΘ[2]
    _snap = rot * snapshot
    # This ignores the difference between ψ and ϕ, because at the 
    # end of the PT, the anisotropic stress is null
    return TΛT(ωs, _snap, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G, kwargs...) * (sin(θ) / 2π)
end

const UnitSphereLowerLeft:: SVector{2, Float64} = SVector{2, Float64}(0., 0.)
const UnitSphereUpperRight:: SVector{2, Float64} = SVector{2, Float64}(2π, π)

function P(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
           ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
           _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
    _integrand(ΦΘ:: SVector{2, Float64}):: Vector{ComplexF64} = integrand(ωs, ΦΘ, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G, kwargs...) 
    return hcubature(_integrand, UnitSphereLowerLeft, UnitSphereUpperRight; kwargs...)
end