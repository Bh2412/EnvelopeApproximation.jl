using EnvelopeApproximation.GravitationalPotentials
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
kvecs = [Vec3(k_horizontal, 0., k_parallel)]
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
η_max = ηs[end]
snapshot = BubblesSnapShot(nucleations, η_max)

# computing Txx, Tyy, Tzz

tds = Vector{TensorDirection}(diagonal)

function T_ij(η:: Float64):: Array{ComplexF64, 2}
    return T_ij(kvecs, current_bubbles(snapshot, η), 1, 1000, ΔV, tds; rtol=1e-3)
end

begin
    Ts = T_ij.(ηs)
    Txx = (x -> x[1]).(Ts)
    Tyy = (x-> x[2]).(Ts)
    Tzz = (x -> x[3]).(Ts)        
end

# comparison Plots For Tij

begin
    p = plot(ηs, Maya_results_df[:, :T_xx], label="Maya", title="Txx")
    plot!(ηs, Txx .|> real, label="Ben")     
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txx_comparison_varying_eta_constant_k.png")

end

begin 
    p = plot(ηs, Maya_results_df[:, :T_yy], label="Maya", title="Tyy")
    plot!(ηs, Tyy .|> real, label="Ben")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tyy_comparison_varying_eta_constant_k.png")
end

begin
    p = plot(ηs, Maya_results_df[:, :T_zz], label="Maya", title="Tzz")
    plot!(ηs, Tzz .|> real, label="Ben")    
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tzz_comparison_varying_eta_constant_k.png")    
end

# comparing ψ

a, G = 1., 1.

function quad_Ψ(η:: Float64):: Vector{ComplexF64}
    return quad_ψ(kvecs, snapshot, η, 1, 100, ΔV, a, G; rtol=1e-2)
end

@time Ψ = @. (x -> x[1])(quad_Ψ(ηs))

Ψ

begin
    n = 60
    p = plot(ηs[1:n], Maya_results_df[!, :Psi][1:n], label="Maya", title="Ψ")
    plot!(ηs[1:n], Ψ[1:n] .|> real, label="Ben")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/psi_comparison_varying_eta_constant_k.png")
end

# Saving the data

d = begin
    d = DataFrame(:η => ηs, :Txx => Txx, :Tyy => Tyy, :Tzz => Tzz, :ψ => Ψ)
    CSV.write("scripts/benchmarks/comparison_to_Maya/gravitational_potential/numeric_varying_eta_constant_k_data.csv", d, delim=';', header=true)
end