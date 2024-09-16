using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.StressEnergyTensor
import LinearAlgebra: norm
using Plots

R = 2.
d = 2.4
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
η_max = 15.
ηs = LinRange(0.5, η_max, 30) |> collect
snapshot = BubblesSnapShot(nucleations, η_max)

k_0 = 2π / (R + d / 2)
ks = LinRange(k_0 / 10, k_0 * 10, 100)
k_vecs = (x -> Vec3(0., 0., x)).(ks)

import EnvelopeApproximation.GravitationalPotentials: quad_ψ as _ψ
@profview ψ = _ψ(k_vecs, snapshot, 5., 10, 10; rtol=1e-2)