using EnvelopeApproximation.GravitationalPotentials: ψ as Ψ, quad_ψ
using CSV, DataFrames
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.StressEnergyTensor
import EnvelopeApproximation.StressEnergyTensor: T_ij, diagonal, TensorDirection
using Plots

# Loading Maya results
Maya_results_df = CSV.read("scripts/benchmarks/comparison_to_Maya/gravitational_potential/varying_eta_fixed_k.csv", DataFrame)
ηs = Maya_results_df[!, :eta]

# Ssetting up the parameters
d = k_parallel = k_horizontal = 1.
ΔV = 1.
kvecs = [Vec3(0., k_horizontal, k_parallel)]
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
η_max = ηs[end]
snapshot = BubblesSnapShot(nucleations, η_max)

# computing Txx, Tyy, Tzz

tds = Vector{TensorDirection}(diagonal)

function T_ij(η:: Float64):: Array{ComplexF64, 2}
    return T_ij(kvecs, current_bubbles(snapshot, η), 20, 20, ΔV, tds; rtol=1e-3)
end

Ts = T_ij.(ηs)
Txx = (x -> x[1]).(Ts)
Tyy = ( x-> x[2]).(Ts)

# comparison Plots

p = plot(ηs, Maya_results_df[:, :T_xx], label="Maya", title="Txx")
plot!(ηs, Txx .|> real, label="Ben")

p = plot(ηs, Maya_results_df[:, :T_yy], label="Maya", title="Tyy")
plot!(ηs, Tyy .|> real, label="Ben")

