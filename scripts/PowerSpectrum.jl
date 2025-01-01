using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import LinearAlgebra: norm
using Plots
using BenchmarkTools
using HCubature
using StaticArrays
import EnvelopeApproximation.GravitationalPotentials: ψ as _ψ, ΦminusΨ
using JLD2

snapshots =  load("evolution_ensemble.jld2", "snapshots")
β = load("evolution_ensemble.jld2", "β")
k_0 = β 
random_k = Vec3(k_0 / 10, 0., 0.)
bubbles = current_bubbles(snapshots[1])
@btime $current_bubbles($snapshots[1])
domes = intersection_domes(bubbles)

# Precompiling
@btime k̂ik̂jTij([random_k], current_bubbles(snapshots[1]); rtol=1e-2)
@profview for _ in 1:1000 k̂ik̂jTij([random_k], current_bubbles(snapshots[1]); rtol=1e-2) end
@btime ψ = _ψ([random_k], snapshots[1], snapshots[1].t; rtol=1e-2)
@btime ΦminusΨ([random_k], snapshots[1], snapshots[1].t; rtol=1e-2)

function kvec(k:: Float64, x:: SVector{2, Float64}):: Vec3
    ϕ, θ = x
    return Vec3(k * sin(θ) * cos(ϕ), k * sin(θ) * sin(ϕ), k * cos(θ)) 
end 

n̂ = SVector{2, Float64}(π / 2, π / 2)
@btime kvec(2., n̂)

V = 1.  # The volume

function integrand(k:: Float64, ΦΘ:: SVector{2, Float64}, 
                   snapshot:: BubblesSnapShot; kwargs...):: ComplexF64
    k_vector = kvec(k, ΦΘ)
    v = ΦminusΨ([k_vector, -k_vector], snapshot, snapshot.t; kwargs...)
    return v[1] * v[2] / V
end

@btime integrand(k_0 / 10, n̂, snapshots[1]; rtol=1e-2)

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
