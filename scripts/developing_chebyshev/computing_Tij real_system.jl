using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor.Δ
import EnvelopeApproximation.GeometricStressEnergyTensor: polar_limits, k̂ik̂jTij, align_ẑ, Ŋ
using EnvelopeApproximation.ChebyshevCFT
import EnvelopeApproximation.ChebyshevCFT: scale, translation
import EnvelopeApproximation.GravitationalPotentials: Ŋ , ΦminusΨ as _legacy_ΦminusΨ, ψ
import EnvelopeApproximation.GeometricStressEnergyTensor: bubble_k̂ik̂jTij_contribution!
import EnvelopeApproximation.ISWPowerSpectrum: P
import LinearAlgebra: norm
using Plots
using BenchmarkTools
using HCubature
using StaticArrays
using IterTools
using QuadGK
using JLD2
using EnvelopeApproximation.Visualization
import Base.*

snapshots = load("evolution_ensemble.jld2", "snapshots")
β = load("evolution_ensemble.jld2", "β")
k_0 = β 
random_k = Vec3(k_0 / 10, 0., 0.)
bubbles = current_bubbles(snapshots[1])
# @btime $current_bubbles($snapshots[1])
domes = intersection_domes(bubbles)
# @btime intersection_domes($bubbles)
const chebyshev_plan:: First3MomentsChebyshevPlan{32} = First3MomentsChebyshevPlan{32}()
const _Δ:: Δ = Δ(1_000)
ks = range(β / 10., β * 10., 100)
V = Vector{ComplexF64}(undef, length(ks))
bubble = bubbles[1]
domes = intersection_domes(bubbles)[1]
ΔV = 1.
bubble_k̂ik̂jTij_contribution!(V, ks, bubble, domes, chebyshev_plan, _Δ; ΔV=ΔV)
# @btime bubble_k̂ik̂jTij_contribution!($V, $ks, $bubble, $domes, $chebyshev_plan, $_Δ; ΔV=ΔV)
# @profview for _ in 1:100 bubble_k̂ik̂jTij_contribution!(V, ks, bubble, domes, chebyshev_plan, _Δ; ΔV=ΔV) end
# plan_16 = First3MomentsChebyshevPlan{16}()
# @btime bubble_k̂ik̂jTij_contribution!($V, $ks, $bubble, $domes, $plan_16, $_Δ; ΔV=$ΔV)
# @profview for _ in 1:1_000 bubble_k̂ik̂jTij_contribution!(V, ks, bubble, domes, plan_16, _Δ; ΔV=ΔV) end
# plot(ks, V .|> real)

V = k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)
# @btime V = k̂ik̂jTij($ks, $bubbles, $chebyshev_plan, $_Δ; ΔV=$ΔV)
# @profview for _ in 1:10  k̂ik̂jTij(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV) end
ks_vecs = [Vec3(0., 0., k) for k in ks]
@time legacy_V = k̂ik̂jTij(ks_vecs, bubbles; rtol=1e-2)
# @btime legacy_V = k̂ik̂jTij($ks_vecs, $bubbles)
# @profview for _ in 1:1_000  k̂ik̂jTij(ks_vecs, bubbles) end

# plot(ks, V .|> real, label="current")
# plot!(ks, legacy_V .|> real, label="legacy")
# plot(ks, V .|> imag)
# plot!(ks, legacy_V .|> imag)

ks_vecs = [Vec3(0., 0., k) for k in ks]
_Ŋ = Ŋ(ks, bubbles, chebyshev_plan, _Δ)
# @btime Ŋ($ks, $bubbles, $chebyshev_plan, $_Δ)
# @time legacy_Ŋ = _legacy_Ŋ(ks_vecs, bubbles; rtol=1e-2)
# @btime legacy_V = ŋ_source($ks_vecs, $bubbles)
# @profview for _ in 1:1_000  k̂ik̂jTij(ks_vecs, bubbles) end

# plot(ks, _Ŋ .|> real, label="current")
# plot!(ks, legacy_Ŋ .|> real, label="legacy")
# plot(ks, V .|> imag)


snapshot = snapshots[1]
_ψ = ψ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2)
# @btime ψ($ks, $snapshot, $chebyshev_plan, $_Δ; rtol=1e-2)
# @profview for _ in 1:1_000 ψ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2) end

# @time _ΦminusΨ = ΦminusΨ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2)
# @profview ΦminusΨ(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-2)
# @time legacy_ΦminusΨ = _legacy_ΦminusΨ(ks_vecs, snapshot, snapshot.t; rtol=1e-2)

# @time ΦminusΨ(ks[1:1], snapshot, chebyshev_plan, _Δ; rtol=1e-2)
# plot(ks, _ΦminusΨ .|> real, label="current")
# plot!(ks, legacy_ΦminusΨ .|> real, label="legacy")

ks = range(β / 10, β * 10, 1_00)

function __P(ks, snapshot)
    plan = First3MomentsChebyshevPlan{32}()
    __Δ = Δ(1000)
    return P(ks, snapshot, plan, __Δ; rtol=1e-2)
end

@time _P = __P(ks, snapshot)

d = Dict()
Threads.@threads for (i, snap) in enumerate(snapshots[1:2])
    d[i] = _p(ks, snap)
end

plot(ks .|> log10, (_P) .|> log10, label=false)
xlabel!("log(k)")
ylabel!("log(P)")
title!("Power Spectrum")
savefig("~/Pictures/FullPowerSpectrum.png")
jldopen("RealSystemPowerSpectrum.jld2", "w") do f
    f["k"] = ks
    f["P"] = _P
end
jldsave("RealSystemPowerSpectrum.jld2", Dict("k" => ks, "P" => _P))

viz(snapshot)