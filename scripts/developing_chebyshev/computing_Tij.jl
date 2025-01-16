using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor.Δ
import EnvelopeApproximation.GeometricStressEnergyTensor: polar_limits, k̂ik̂jTij, align_ẑ
using EnvelopeApproximation.ChebyshevCFT
import EnvelopeApproximation.ChebyshevCFT: scale, translation
import EnvelopeApproximation.GravitationalPotentials: Ŋ as _legacy_Ŋ, ΦminusΨ as _legacy_ΦminusΨ
import LinearAlgebra: norm
using Plots
using BenchmarkTools
using HCubature
using StaticArrays
using IterTools
using QuadGK
import Base.*

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
const chebyshev_plan:: First3MomentsChebyshevPlan{32} = First3MomentsChebyshevPlan{32}()
const _Δ:: Δ = Δ(1_000)

function bubble_k̂ik̂jTij_contribution!(V:: AbstractVector{ComplexF64},
                                      ks:: AbstractVector{Float64}, 
                                      bubble:: Bubble, 
                                      domes:: Vector{IntersectionDome}, 
                                      chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                                      _Δ:: Δ; 
                                      ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert length(V) == length(ks) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _Δ(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3])
            _, c1, c2 = fourier_mode(k * bubble.radius, chebyshev_plan, s, t)
            V[i] += c2 * (ΔV * (bubble.radius ^ 3) / 3) * e # ∂_iφ∂_jφ contribution
            V[i] -= c1 * (-im * ΔV) * e / k * (bubble.radius ^ 2) # potential contribution
        end
    end
    return V
end

ks = range(k_0 / 10., k_0 * 10., 100)
V = Vector{ComplexF64}(undef, length(ks))
bubble = bubbles[1]
domes = intersection_domes(bubbles)[1]
ΔV = 1.
bubble_k̂ik̂jTij_contribution!(V, ks, bubble, domes, chebyshev_plan, _Δ; ΔV=ΔV)
plan_16 = First3MomentsChebyshevPlan{16}()
@btime bubble_k̂ik̂jTij_contribution!($V, $ks, $bubble, $domes, $plan_16, $_Δ; ΔV=$ΔV)
@profview for _ in 1:1_000 bubble_k̂ik̂jTij_contribution!(V, ks, bubble, domes, plan_16, _Δ; ΔV=ΔV) end
# plot(ks, V .|> real)

function k̂ik̂jTij(ks:: AbstractVector{Float64}, 
                 bubbles:: AbstractVector{Bubble}, 
                 chebyshev_plan:: First3MomentsChebyshevPlan{N},
                 _Δ:: Δ;
                 ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    V = zeros(ComplexF64, length(ks))
    domes = intersection_domes(bubbles)
    for (bubble_index, _domes) in domes
        bubble_k̂ik̂jTij_contribution!(V, ks, bubbles[bubble_index], _domes, 
                                     chebyshev_plan, _Δ; ΔV=ΔV)
    end
    return V
end

V = k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)
# @btime V = k̂ik̂jTij($ks, $bubbles, $chebyshev_plan, $_Δ; ΔV=$ΔV)
# @profview for _ in 1:1_000  k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV) end
ks_vecs = [Vec3(0., 0., k) for k in ks]
legacy_V = k̂ik̂jTij(ks_vecs, bubbles)
# @btime legacy_V = k̂ik̂jTij($ks_vecs, $bubbles)
# @profview for _ in 1:1_000  k̂ik̂jTij(ks_vecs, bubbles) end

# plot(ks, V .|> real, label="current")
# plot!(ks, legacy_V .|> real, label="legacy")
# plot(ks, V .|> imag)

function ψ_source(ks:: AbstractVector{Float64}, 
                  bubbles:: AbstractVector{Bubble}, 
                  chebyshev_plan:: First3MomentsChebyshevPlan{N},
                  _Δ:: Δ;
                  ΔV:: Float64 = 1., 
                  a:: Float64 = 1.,
                  G:: Float64 = 1.) where N
    return (4π * a ^ 2 * G) * k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)                  
end

function ψ(ks:: AbstractVector{Float64}, 
           snapshot:: BubblesSnapShot,
           chebyshev_plan:: First3MomentsChebyshevPlan{N},
           _Δ:: Δ;
           ΔV:: Float64 = 1., 
           a:: Float64 = 1.,
           G:: Float64 = 1., 
           kwargs...) where N
    t = snapshot.t
    f(τ:: Float64):: Vector{ComplexF64} = ψ_source(ks, current_bubbles(snapshot, τ), 
                                                   chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G) * (t - τ)            
    return quadgk(f, 0., t; kwargs...)[1]
end

function bubble_Ŋ_contribution!(V:: AbstractVector{ComplexF64},
                                ks:: AbstractVector{Float64}, 
                                bubble:: Bubble, 
                                domes:: Vector{IntersectionDome}, 
                                chebyshev_plan:: First3MomentsChebyshevPlan{N}, 
                                _Δ:: Δ; 
                                ΔV:: Float64 = 1.):: Vector{ComplexF64} where N
    @assert length(V) == length(ks) "The output vector must be of the same length of the input k vector"
    _polar_limits = polar_limits(bubble.radius, domes)
    for (μ1, μ2) in partition(_polar_limits, 2, 1)
        s, t = scale(μ1, μ2), translation(μ1, μ2)
        chebyshev_coeffs!(μ -> _Δ(μ, bubble, domes), μ1, μ2, chebyshev_plan)
        for (i, k) in enumerate(ks)
            e = cis(-k * bubble.center.coordinates[3])
            c0, _, c2 = fourier_mode(k * bubble.radius, chebyshev_plan, s, t)
            V[i] += (c2 - (1. / 3) * c0) * (ΔV * (bubble.radius ^ 3) / 3) * e # ∂_iφ∂_jφ contribution
        end
    end
    return V
end

function Ŋ(ks:: AbstractVector{Float64}, 
           bubbles:: AbstractVector{Bubble}, 
           chebyshev_plan:: First3MomentsChebyshevPlan{N},
           _Δ:: Δ;
           ΔV:: Float64 = 1.,
           a:: Float64 = 1.,
           G:: Float64 = 1.,) where N
    V = zeros(ComplexF64, length(ks))
    domes = intersection_domes(bubbles)
    for (bubble_index, _domes) in domes
        bubble_Ŋ_contribution!(V, ks, bubbles[bubble_index], _domes, 
                               chebyshev_plan, _Δ; ΔV=ΔV)
    end
    c = -12π * G * a ^ 2
    return @. V * c / (ks ^ 2)
end

ks_vecs = [Vec3(0., 0., k) for k in ks]
_Ŋ = Ŋ(ks, bubbles, chebyshev_plan, _Δ)
# @btime Ŋ($ks, $bubbles, $chebyshev_plan, $_Δ)
legacy_Ŋ = _legacy_Ŋ(ks_vecs, bubbles)
# @btime legacy_V = ŋ_source($ks_vecs, $bubbles)
# @profview for _ in 1:1_000  k̂ik̂jTij(ks_vecs, bubbles) end

# plot(ks, _Ŋ .|> real, label="current")
# plot!(ks, legacy_Ŋ .|> real, label="legacy")
# plot(ks, V .|> imag)


_ψ = ψ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2)
# @btime ψ($ks, $snapshot, $chebyshev_plan, $_Δ; rtol=1e-2)
# @profview for _ in 1:1_000 ψ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2) end

function ΦminusΨ(ks:: AbstractVector{Float64}, 
                 snapshot:: BubblesSnapShot,
                 chebyshev_plan:: First3MomentsChebyshevPlan{N},
                 _Δ:: Δ;
                 ΔV:: Float64 = 1., 
                 a:: Float64 = 1.,
                 G:: Float64 = 1., 
                 kwargs...) where N
    _ψ = ψ(ks, snapshot, chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G, kwargs...)
    #_Ŋ = Ŋ(ks, current_bubbles(snapshot), chebyshev_plan, _Δ; ΔV=ΔV, a=a, G=G)
    return @. - 2 * _ψ
end


_ΦminusΨ = ΦminusΨ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2)
legacy_ΦminusΨ = _legacy_ΦminusΨ(ks_vecs, snapshot, snapshot.t; rtol=1e-2)

# plot(ks, _ΦminusΨ .|> real, label="current")
# plot!(ks, legacy_ΦminusΨ .|> real, label="legacy")

*(rot:: SMatrix{3, 3, Float64}, p:: Point3):: Point3 = Point3(rot * p.coordinates)

function *(rot:: SMatrix{3, 3, Float64}, snapshot:: BubblesSnapShot):: BubblesSnapShot
    new_nucleations = [(time=nuc.time, site=rot * nuc.site) for nuc in snapshot.nucleations]
    return BubblesSnapShot(new_nucleations, snapshot.t, snapshot.radial_profile)
end

function n̂(x:: SVector{2, Float64}):: Vec3
    ϕ, θ = x
    return Vec3((sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))) 
end 

function integrand(ks:: AbstractVector{Float64}, 
                   ΦΘ:: SVector{2, Float64}):: Vector{ComplexF64}
    rot = align_ẑ(n̂(ΦΘ))
    _snap = rot * snapshot
    return ΦminusΨ(ks, _snap, chebyshev_plan, _Δ; rtol=1e-2) .|> abs2
end

# @btime integrand(ks, SVector(0., π /2))

ll = SVector{2, Float64}(0., 0.)
ur = SVector{2, Float64}(2π, π / 2)

function P(ks:: AbstractVector{Float64}):: Vector{Float64}
    return hcubature(x -> integrand(ks, x), ll, ur; rtol=1e-2)[1]
end

ks = range(10 ^ 0.4 ,  10 ^ 0.6, 1_000)

@time _P = P(ks)

plot(ks .|> log10, (_P) .|> log10)