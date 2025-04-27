begin
using Bessels
import Bessels: besselj!
import EnvelopeApproximation.ChebyshevCFT: values!, chebyshev_coeffs!, fourier_mode, scale, translation, multiplication_weights, chebyshevpoints, First3MomentsChebyshevPlan, inverse_chebyshev_weight, inverse_u
using FastTransforms
using BenchmarkTools
using LinearAlgebra
using Test
using StaticArrays
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
import EnvelopeApproximation.GeometricStressEnergyTensor: ∫_ϕ, upper_right, _buffers, PeriodicInterval, IntersectionDome, align_ẑ, intersection_domes, polar_limits, ring_domes_complement_intersection!
using EnvelopeApproximation.BubblesEvolution
import EnvelopeApproximation.BubblesEvolution: BallSpace, BubblesSnapShot
import EnvelopeApproximation.ISWPowerSpectrum: n̂
using EnvelopeApproximation.GravitationalWaves: x̂_ix̂_j, VectorChebyshevPlan
using QuadGK
using HCubature
using StaticArrays
using IterTools
import Base.rand
import Base.*
end

# @btime fourier_modes($ks, $vector_f, -1., 1., $vector_chebyshev_plan)
# @btime fourier_modes($ks, $f, -1., 1., $moments_chebyshev_plan)

# @profview for _ in 1:10_000 fourier_modes(ks, vector_f, -1., 1., vector_chebyshev_plan) end
# @profview for _ in 1:10_000 fourier_modes(ks, f, -1., 1., moments_chebyshev_plan) end

# function ∫_ϕ_x̂_ix̂_j(μ:: Float64, p:: PeriodicInterval):: NTuple{6, Float64}
#     ϕ1, ϕ2 = p.ϕ1, p.ϕ1 + p.Δ
#     s2 = 1 - μ ^ 2
#     s = sqrt(s2)
#     return s2 * ((1 / 2) * (ϕ2 - ϕ1) + (1/4) * (sin(2ϕ2) - sin(2ϕ1))), s2 * (1 / 4) * (cos(2ϕ1) - cos(2ϕ2)), (μ * s) * (sin(ϕ2) - sin(ϕ1)), s2 * ((1/2) * (ϕ2 - ϕ1) - (1/4) * (sin(2ϕ2) - sin(2ϕ1))), μ * s * (cos(ϕ1) - cos(ϕ2)), μ ^ 2 * (ϕ2 - ϕ1) 
# end

# begin 

#     function rand(p:: Type{PeriodicInterval}):: PeriodicInterval
#         x1, x2 = 2π .* rand(Float64, 2)
#         return PeriodicInterval(x1, x2)
#     end

#     function rand(p:: Type{PeriodicInterval}, n:: Int64):: Vector{PeriodicInterval}
#         V = Vector{PeriodicInterval}(undef, n)
#         for i in 1:n
#             V[i] = rand(p)
#         end
#         return V
#     end

#     ps = rand(PeriodicInterval, 1_000)
#     μs = range(-1., 1., 1_000)
#     @test all([all(∫_ϕ_x̂_ix̂_j(μ, p) .≈ ∫_ϕ(upper_right, μ, p.ϕ1, p.ϕ1 + p.Δ)) for μ in μs for p in ps])

# end

# begin
    

# struct x̂_ix̂_j
#     arcs_buffer:: Vector{PeriodicInterval}
#     limits_buffer:: Vector{Tuple{Float64, Float64}}
#     intersection_buffer:: Vector{PeriodicInterval}
# end

# x̂_ix̂_j(n:: Int64) = x̂_ix̂_j(_buffers(n)...)


# function (f:: x̂_ix̂_j)(μ:: Float64, bubble:: Bubble, 
#                       intersection_domes:: Vector{IntersectionDome}):: MVector{6, Float64}
#     V = zeros(MVector{6, Float64})
#     periodic_intervals = ring_domes_complement_intersection!(μ, bubble.radius, intersection_domes, 
#                                                              f.arcs_buffer, f.limits_buffer, f.intersection_buffer)
#     @inbounds for interval in periodic_intervals
#         V .+= ∫_ϕ_x̂_ix̂_j(μ, interval)
#     end
#     return V
# end

# function bubble_Tij_contribution!(V:: AbstractMatrix{ComplexF64},
#                                   ks:: AbstractVector{Float64}, 
#                                   bubble:: Bubble, 
#                                   domes:: Vector{IntersectionDome}, 
#                                   chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#                                   _x̂_ix̂_j:: x̂_ix̂_j; 
#                                   ΔV:: Float64 = 1.) where N
#     @assert size(V) == (length(ks), 6) "The output vector must be of the same length of the input k vector"
#     _polar_limits = polar_limits(bubble.radius, domes)
#     @inbounds for (μ1, μ2) in partition(_polar_limits, 2, 1)
#         s, t = scale(μ1, μ2), translation(μ1, μ2)
#         chebyshev_coeffs!(μ -> _x̂_ix̂_j(μ, bubble, domes), μ1, μ2, chebyshev_plan)
#         @inbounds for (i, k) in enumerate(ks)
#             e = cis(-k * bubble.center.coordinates[3]) * (ΔV * (bubble.radius ^ 3) / 3)
#             @. V[i, :] += e * $fourier_mode(k, chebyshev_plan, s, t) # ∂_iφ∂_jφ contribution
#         end
#     end
# end

# function Tij(ks:: AbstractVector{Float64}, 
#              bubbles:: AbstractVector{Bubble}, 
#              ball_space:: BallSpace,
#              chebyshev_plan:: VectorChebyshevPlan{N, 6},
#              _x̂_ix̂_j:: x̂_ix̂_j;
#              ΔV:: Float64 = 1.):: Matrix{ComplexF64} where N
#     V = zeros(ComplexF64, length(ks), 6)
#     domes = intersection_domes(bubbles, ball_space)
#     @inbounds for (bubble_index, _domes) in domes
#         bubble_Tij_contribution!(V, ks, bubbles[bubble_index], _domes, 
#                                  chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#     end
#     return V
# end

# const symmetric_tensor_indices:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2=> (1, 2), 3=> (1, 3), 4 =>(2, 2), 5 =>(2, 3), 6 => (3, 3))
# const inverse_symmetric_tensor_indices:: Dict{Tuple{Int, Int}, Int} = Dict(zip(values(symmetric_tensor_indices), keys(symmetric_tensor_indices)))

# function symmetric_dot(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
#     r = 0.
#     for ĩ in 1:6
#         (i, j) = symmetric_tensor_indices[ĩ]
#         (i == j) && (r += (T1[ĩ])' * T2[ĩ]); continue
#         r += 2 * (T1[ĩ])' * T2[ĩ]
#     end
#     return r
# end

# function δ(T:: AbstractVector{ComplexF64}):: ComplexF64
#     return T[1] + T[4] + T[6]
# end

# function zz(T:: AbstractVector{ComplexF64})
#     return T[6]
# end

# function Λ(T1:: AbstractVector{ComplexF64}, T2:: AbstractVector{ComplexF64}):: ComplexF64
#     r = 0.
#     r += symmetric_dot(T1, T2)
#     r += (-2) * @views (T1[[3, 5, 6]]' * T2[[3, 5, 6]])
#     zz1 = zz(T1)'
#     zz2 = zz(T2)
#     δ1 = δ(T1)'
#     δ2 = δ(T2)
#     r += (1. / 2) * zz1 * zz2
#     r += (-1. / 2) * δ1 * δ2
#     r += (1. / 2) * δ1 * zz2
#     r += (1. / 2) * zz1 * δ2
#     return r
# end

# function Λ(T:: AbstractVector{ComplexF64}):: Float64
#     return Λ(T, T)
# end

# function *(rot:: SMatrix{3, 3, Float64}, bubble:: Bubble)
#     return Bubble(rot * bubble.center, bubble.radius)
# end

# function _TΛT(ωs:: AbstractVector{Float64}, bubbles:: Bubbles, 
#     ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#     _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1.):: Vector{Float64} where N
#     # Eq. 15 in Kosowsky and Turner
#     T = Tij(ωs, bubbles, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#     return @. Λ($eachrow(T)) * (2G * (ωs ^ 2))
# end

# function _TΛT(t:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#               ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#               _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1.):: Vector{Float64} where N
#     # Eq. 15 in Kosowsky and Turner
#     return _TΛT(ωs, current_bubbles(snapshot, t), ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G)
# end

# # Eq. 16 in "gravitational waves from bubble collisions: analytic derivation".
# function Directional_Π(_n̂:: Vec3, t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#                        ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#                        _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
#     _snap = align_ẑ(_n̂) * snapshot
#     bubbles1 = current_bubbles(_snap, t1)
#     bubbles2 = current_bubbles(_snap, t2)
#     T1 = Tij(ωs, bubbles1, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#     T2 = Tij(ωs, bubbles2, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#     return @. Λ($eachrow(T1), $eachrow(T2)) 
# end

# function Π(t1:: Float64, t2:: Float64, ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#            ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#            _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., kwargs...):: Tuple{Vector{ComplexF64}, Float64} where N
#     function f(_n̂:: SVector{2, Float64}):: Vector{ComplexF64}
#         return Directional_Π(n̂(_n̂), t1, t2, ωs, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#     end
#     v, err = hcubature(f, SVector(0., 0.,), SVector(2π, π); kwargs...)
#     return v ./ 4π, err / 4π
# end

# function TΛT(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#              ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#              _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
#     # Eq. 15 in Kosowsky and Turner
#     function f(t:: Float64):: Vector{ComplexF64}
#         return @. (cis(ωs * t) / 2π) * $_TΛT(t, ωs, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G)
#     end
#     return quadgk(f, 0., snapshot.t; kwargs...)[1]
# end

# function integrand(ωs:: AbstractVector{Float64}, 
#                    ΦΘ:: SVector{2, Float64}, 
#                    snapshot:: BubblesSnapShot,
#                    ball_space:: BallSpace, 
#                    chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#                    _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., 
#                    G:: Float64 = 1., kwargs...):: Vector{ComplexF64} where N
#     rot = align_ẑ(n̂(ΦΘ))
#     θ = ΦΘ[2]
#     _snap = rot * snapshot
#     # This ignores the difference between ψ and ϕ, because at the 
#     # end of the PT, the anisotropic stress is null
#     return TΛT(ωs, _snap, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G, kwargs...) * sin(θ) # Note that this is not an average over sky point
# end

# const UnitSphereLowerLeft:: SVector{2, Float64} = SVector{2, Float64}(0., 0.)
# const UnitSphereUpperRight:: SVector{2, Float64} = SVector{2, Float64}(2π, π)

# function separate_P(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#                     ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#                     _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...):: Tuple{Vector{ComplexF64}, Float64} where N
#     int(ΦΘ) = integrand(ωs, ΦΘ, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV, G=G, kwargs...)
#     return hcubature(int, UnitSphereLowerLeft, UnitSphereUpperRight; kwargs...)
# end

# *(rot:: SMatrix{3, 3, Float64}, bubbles:: Bubbles) = map(x -> rot * x, bubbles)

# function _integrand(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#     ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#     _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...) where N
# function integrand(t1t2ΦΘ:: SVector{4, Float64}):: Vector{ComplexF64}
#  t1, t2, ϕ, θ = t1t2ΦΘ
#  exps = sin(θ) * (@. cis(ωs * (t1 + t2))) / (4 * π ^ 2)
#  rot = align_ẑ(n̂(ϕ, θ))
#  bubbles1 = rot * current_bubbles(snapshot, t1)
#  bubbles2 = rot * current_bubbles(snapshot, t2)
#  T1 = Tij(ωs, bubbles1, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#  T2 = Tij(ωs, bubbles2, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#  return @. Λ($eachrow(T1), $eachrow(T2)) * (2G * (ωs ^ 2)) * exps
# end
# return integrand
# end

# function P(ωs:: AbstractVector{Float64}, snapshot:: BubblesSnapShot, 
#            ball_space:: BallSpace, chebyshev_plan:: VectorChebyshevPlan{N, 6}, 
#            _x̂_ix̂_j:: x̂_ix̂_j; ΔV:: Float64 = 1., G:: Float64 = 1., kwargs...):: Tuple{Vector{ComplexF64}, Float64} where N
#     function integrand(t1t2ΦΘ:: SVector{4, Float64}):: Vector{ComplexF64}
#         t1, t2, ϕ, θ = t1t2ΦΘ
#         exps = sin(θ) * (@. cis(ωs * (t1 + t2))) / (4 * π ^ 2)
#         rot = align_ẑ(n̂(ϕ, θ))
#         bubbles1 = rot * current_bubbles(snapshot, t1)
#         bubbles2 = rot * current_bubbles(snapshot, t2)
#         T1 = Tij(ωs, bubbles1, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#         T2 = Tij(ωs, bubbles2, ball_space, chebyshev_plan, _x̂_ix̂_j; ΔV=ΔV)
#         return @. Λ($eachrow(T1), $eachrow(T2)) * (2G * (ωs ^ 2)) * exps
#     end
#     return hcubature(integrand, SVector(0., 0., 0., 0.), SVector(snapshot.t, snapshot.t, 2π, π); kwargs...)
# end

# end

begin 
    d = 1.
    R = 1.2 * d  # Like Kosowsky and Turner
    nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
    snapshot = BubblesSnapShot(nucleations, R)
    bubbles = current_bubbles(snapshot)
    k_0 = 2π / (R + d / 2)
    ks = logrange(k_0 / 100, k_0 * 10, 100)
    chebyshev_plan = VectorChebyshevPlan{32, 6}()
    _x̂_ix̂_j = x̂_ix̂_j(4)
    ball_space = BallSpace(2.1R + d, Point3(0., 0., 0.))
end

begin
    domes = intersection_domes(bubbles)
    _domes = domes[1]
    bubble = bubbles[1]
    μc = _domes[1].h / R
    μs = range(-1., 1., 1_000)
    f(μ) = begin
        if μ <= μc
            return [π * (1 - μ ^ 2), 0., 0., π * (1 - μ ^ 2), 0., 2π * μ ^ 2]
        else
            return zeros(6)
        end
    end
    @test _x̂_ix̂_j.(μs, (bubble, ), (_domes, )) ≈ f.(μs) 
end

# begin
#     t1 = snapshot.t / 2
#     t2 = snapshot.t * (4/5)
#     ωs = range(0., 10., 100)
#     @time Π(t1, t2, ωs, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; rtol=1e-3)
# end

# begin
#     @btime $_x̂_ix̂_j(0., $bubble, $_domes)
#     # @profview for _ in 1:10_000_000  _x̂_ix̂_j(0., bubble, _domes) end
# end

# begin
#     ts = range(0., snapshot.t, 1_000)
#     snap = align_ẑ(EnvelopeApproximation.BubbleBasics.Vec3(1., 0., 0.)) * snapshot
#     _TΛTs = map(ts) do t
#         _TΛT(t, ks, snap, ball_space, chebyshev_plan, _x̂_ix̂_j) .* cis.(ks * t)
#     end
#     m = Matrix{ComplexF64}(undef, length(ks), length(ts))
#     for (i, v) in enumerate(_TΛTs)
#         m[:, i] = v
#     end
#     using CairoMakie
#     fig=Figure()
#     ax = Axis(fig[1, 1], xlabel="ks", )
#     heatmap!(ax, ks[ks .> 10.], ts, abs.(m[ks .> 10.,:]))
#     fig
# end

# begin
#     t = snapshot.t
#     ks = range(k_0 / 100, 10k_0, 100)
#     # @btime P($t, $ks, $snapshot, $ball_space, $chebyshev_plan, $_x̂_ix̂_j)
#     @time p = P(ks, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; rtol=1e-3)
# end

# using CairoMakie

# begin
#     using CairoMakie
#     fig = Figure()
#     ax = Axis(fig[1, 1], title="2 Bubble GW energy density", xlabel="ω", ylabel=L"$\frac{dE}{dω}$", 
#               limits=((0., 10.), (0., 1.)))
#     lines!(ax, ks, abs.(p[1]))
#     vlines!(ax, [3.8 * 1 / t])
#     save("scripts/GW/GW_energy_density_2_bubbles.png", fig)
#     fig
# end



# @time TΛT(ks, snap, ball_space, chebyshev_plan, _x̂_ix̂_j; rtol=1e-2)
# # @time P(ks, snapshot, ball_space, chebyshev_plan, _x̂_ix̂_j; rtol=1e-3)
