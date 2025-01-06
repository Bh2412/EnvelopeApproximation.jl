using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor: intersection_domes, BubbleArck̂ik̂j∂iφ∂jφ, ∫_ϕ
import EnvelopeApproximation.GeometricStressEnergyTensor: align_ẑ, polar_limits, ring_dome_intersection
import EnvelopeApproximation.GeometricStressEnergyTensor: fourier_mode
import EnvelopeApproximation.GeometricStressEnergyTensor: _buffers, ring_domes_intersection!
import LinearAlgebra: norm
using Plots
using BenchmarkTools
using HCubature
using StaticArrays
import EnvelopeApproximation.GravitationalPotentials: ψ as _ψ, ΦminusΨ
using JLD2
using IterTools
using QuadGK
using ApproxFun

snapshots =  load("evolution_ensemble.jld2", "snapshots")
β = load("evolution_ensemble.jld2", "β")
k_0 = β 
random_k = Vec3(k_0 / 10, 0., 0.)
bubbles = current_bubbles(snapshots[1])
# @btime $current_bubbles($snapshots[1])
domes = intersection_domes(bubbles)
bubble = bubbles[1]
bubble_domes = domes[1]
int = BubbleArck̂ik̂j∂iφ∂jφ(bubble.radius, bubble_domes)
∫_ϕ(int, -0.9)
_polar_limits = polar_limits(bubble.radius, bubble_domes)

# @btime fourier_mode(int, k_0 * bubble.radius)
# @profview for _ in 1:1000 fourier_mode(int, k_0 * bubble.radius) end

function _fourier_mode(κ:: Float64)
    return quadgk_count(μ -> cis(-μ * κ) * ∫_ϕ(int, μ), _polar_limits[[1, 2]]; rtol=1e-2)
end

# @btime _fourier_mode(k_0 * bubble.radius)

struct Δ
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

Δ(n:: Int64) = Δ(_buffers(n)...)

function (δ:: Δ)(μ:: Float64, bubble:: Bubble, 
                 intersection_domes:: Vector{IntersectionDome}):: Float64
    periodic_intervals = ring_domes_intersection!(μ, bubble.radius, intersection_domes, 
                                                  δ.arcs_buffer, δ.limits_buffer, δ.intersection_buffer)
    return sum((p.Δ for p in periodic_intervals), init=0.)
end

const _Δ:: Δ = Δ(length(bubbles))
const b:: Bubble = bubble
const bdomes:: Vector{IntersectionDome} = bubble_domes

function bubble_Δ(μ:: Float64) :: Float64
    return _Δ(μ, b, bdomes)
end

# μ = 0.5
# for (μ1, μ2) in partition(_polar_limits, 2, 1)
#     μ_c = (μ1 + μ2) / 2
#     @show μ_c
#     value = @btime $bubble_Δ($μ_c)
#     @show value
# end

_μs = -1. : 0.01 : 1.
plot(_μs, bubble_Δ.(_μs))
# Precompiling
# @btime k̂ik̂jTij([random_k], current_bubbles(snapshots[1]); rtol=1e-2)
# @profview for _ in 1:1000 k̂ik̂jTij([random_k], current_bubbles(snapshots[1]); rtol=1e-2) end
# @btime ψ = _ψ([random_k], snapshots[1], snapshots[1].t; rtol=1e-2)
# @btime ΦminusΨ([random_k], snapshots[1], snapshots[1].t; rtol=1e-2)

function kvec(k:: Float64, x:: SVector{2, Float64}):: Vec3
    ϕ, θ = x
    return Vec3(k * sin(θ) * cos(ϕ), k * sin(θ) * sin(ϕ), k * cos(θ)) 
end 

n̂ = SVector{2, Float64}(π / 2, π / 2)
@btime kvec(2., n̂)

V = 1.  # The volume

function integrand(k:: Float64, ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot; kwargs...):: Float64
    k_vector = kvec(k, ΦΘ)
    v = ΦminusΨ([k_vector], snapshot, snapshot.t; kwargs...)
    return abs(v[1]) / V
end

@time integrand(k_0 / 10, n̂, snapshots[1]; rtol=1e-2)

ll = SVector{2, Float64}(0., 0.)
ur = SVector{2, Float64}(2π, π)

function P(k:: Float64, snapshot:: BubblesSnapShot; kwargs...):: ComplexF64
    return hcubature(x -> integrand(k, x, snapshot; kwargs...), ll, ur; kwargs...)[1]
end

P(k_0 / 10, snapshots[1]; rtol=1e-2)

ks = LinRange(k_0 / 10, k_0, 1)
PS = P.(ks; rtol=1e-2)
plot(ks, ks .^ 4 .* PS .|> real)
plot!(ks, (@. 1 / (ks ^ 2)) .|> log)
