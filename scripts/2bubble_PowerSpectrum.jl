using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import LinearAlgebra: norm
using Plots
using BenchmarkTools
using HCubature
using StaticArrays

R = 2.
d = 2.4
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
snapshot = BubblesSnapShot(nucleations, R)
bubbles = current_bubbles(snapshot)
k_0 = 2π / (R + d / 2)
ks = LinRange(k_0 / 10, k_0 * 10, 100)
k_vecs = (x -> Vec3(0., 0., x)).(ks)
k_0
ks
ts = LinRange(0., R, 10) |> collect
η_PT = R

# Precompiling

# @btime k̂ik̂jTij(k_vecs[[1]], bubbles; rtol=1e-2)
# @profview for _ in 1:100_000 k̂ik̂jTij(k_vecs[[1]], bubbles; rtol=1e-2) end


@btime potential_integral(k_vecs[[1]], bubbles;rtol=1e-3)


import EnvelopeApproximation.GravitationalPotentials: ψ as _ψ, ΦminusΨ


# Precompiling
# @btime ψ = _ψ(k_vecs[[100]], snapshot, R; rtol=1e-2)
# @profview for _ in 1:1_000 _ψ(k_vecs[[100]], snapshot, R; rtol=1e-2) end
# @btime ΦminusΨ(k_vecs[[1]], snapshot, R; rtol=1e-2)

function kvec(k:: Float64, x:: SVector{2, Float64}):: Vec3
    ϕ, θ = x
    return Vec3((k * sin(θ) * cos(ϕ), k * sin(θ) * sin(ϕ), k * cos(θ))) 
end 

n̂ = SVector{2, Float64}(π / 2, π / 2)
_k = 2.
# @btime kvec($_k, $n̂)
# @btime k̂ik̂jTij([kvec(ks[1], n̂)], bubbles; rtol=1e-2)

# Studying why _ψ takes so long
# @btime ψ = _ψ([kvec(ks[1], n̂)], snapshot, R; rtol=1e-2)
# @profview for _ in 1:100 ψ = _ψ([kvec(ks[1], n̂)], snapshot, R; rtol=1e-2) end
# @profview for _ in 1:100 ΦminusΨ(k_vecs[[1]], snapshot, R; rtol=1e-2) end

# Studying why computing _ψ takes so long
function source(k:: Vec3, t:: Float64):: ComplexF64
    bubbles = current_bubbles(snapshot, t)
    return k̂ik̂jTij([k], bubbles; rtol=1e-2)[1] * (η_PT - t)
end


V = 1.  # The volume

function integrand(k:: Float64, ΦΘ:: SVector{2, Float64}; kwargs...):: ComplexF64
    k_vector = kvec(k, ΦΘ)
    v = ΦminusΨ([k_vector, -k_vector], snapshot, η_PT; kwargs...)
    return v[1] * v[2] / V
end

# @btime integrand(ks[1], n̂; rtol=1e-2)

ll = SVector{2, Float64}(0., 0.)
ur = SVector{2, Float64}(2π, π)

function P(k:: Float64; kwargs...):: ComplexF64
    return hcubature(x -> integrand(k, x; kwargs...), ll, ur; kwargs...)[1]
end

@btime P(ks[100]; rtol=1e-2)
@profview P(ks[1]; rtol=1e-2)
@time P(ks[ks .<= k_0][end]; rtol=1e-2)

ks = LinRange(k_0 / 10, k_0, 100)
PS = P.(ks; rtol=1e-2)
plot(ks, PS .|> real)
plot!(ks, (@. 1 / (ks ^ 2)) .|> log)
