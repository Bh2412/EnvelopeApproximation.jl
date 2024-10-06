using EnvelopeApproximation.GravitationalPotentials
using CSV, DataFrames
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.StressEnergyTensor
import EnvelopeApproximation.StressEnergyTensor: T_ij, diagonal, TensorDirection, upper_right
import EnvelopeApproximation.GravitationalPotentials: ψ_source, Ŋ, Φ
using Plots

# Loading Maya results
Maya_results_df = CSV.read("scripts/benchmarks/comparison_to_Maya/gravitational_potential/varying_eta_fixed_k_data_Maya.csv", DataFrame)
ηs = Maya_results_df[!, :eta]

# Ssetting up the parameters
d = k_parallel = k_horizontal = 1.
ΔV = 1.
kvecs = [Vec3(k_horizontal, 0., k_parallel)]
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
η_max = ηs[end]
snapshot = BubblesSnapShot(nucleations, η_max)

# computing Txx, Tyy, Tzz

tds = Vector{TensorDirection}(upper_right)

function Tij(η:: Float64):: Array{ComplexF64, 2}
    return T_ij(kvecs, current_bubbles(snapshot, η), 1, 1000, ΔV, tds; rtol=1e-3)
end

function ψsource(η:: Float64)
    return ψ_source(kvecs, snapshot, η, 1, 1000, ΔV; rtol=1e-3)
end

begin
    Ts = Tij.(ηs)
    Txx = (x -> x[1]).(Ts)
    Tyy = (x-> x[2]).(Ts)
    Tzz = (x -> x[3]).(Ts)
    Txy = (x -> x[4]).(Ts)
    Txz = (x -> x[5]).(Ts)
    Tyz = (x -> x[6]).(Ts)        
    ψs = @. (x -> x[1])(ψsource(ηs))
end

# comparison Plots For Tij

begin
    p = plot(ηs, Maya_results_df[:, :T_xx], label="Maya", title="Txx")
    plot!(ηs, Txx .|> real, label="Ben")     
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txx_comparison_varying_eta_constant_k.png")
    display(p)
end

begin 
    p = plot(ηs, Maya_results_df[:, :T_yy], label="Maya", title="Tyy")
    plot!(ηs, Tyy .|> real, label="Ben")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tyy_comparison_varying_eta_constant_k.png")
    display(p)
end

begin
    p = plot(ηs, Maya_results_df[:, :T_zz], label="Maya", title="Tzz")
    plot!(ηs, Tzz .|> real, label="Ben")    
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tzz_comparison_varying_eta_constant_k.png")    
    display(p)
end

begin
    p = plot(ηs, Maya_results_df[:, :T_zx], label="Maya", title="Tzx")
    plot!(ηs, Txz .|> real, label="Ben")    
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txz_comparison_varying_eta_constant_k.png")    
    display(p)
end

# comparing ψ

a, G = 1., 1.

function quad_Ψ(η:: Float64):: Vector{ComplexF64}
    return quad_ψ(kvecs, snapshot, η, 1, 100, ΔV, a, G; rtol=1e-2)
end

@time Ψ = @. (x -> x[1])(quad_Ψ(ηs))

Ψ

ŋ = (T -> Ŋ(kvecs, T, tds, a, G)).(Ts) .|> x -> x[1]

ϕ = Φ(ŋ, Ψ)

begin
    p = plot(ηs, ψs .|> real, label="ψ``", xlabel="η")
    plot!(ηs, Ψ .|> real, label="ψ")  
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/gravitational_potential_source_varying_eta_constant_k.png")    
    display(p)
end

# plot

begin
    n = 120
    p = plot(ηs[1:n], Maya_results_df[!, :Psi][1:n], label="Maya", title="Ψ", xlabel="η")
    plot!(ηs[1:n], Ψ[1:n] .|> real, label="Ben")
    display(p)
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/psi_comparison_varying_eta_constant_k.png")
end

# difference plot

begin
    n = 60
    p = plot(ηs[1:n], (Maya_results_df[!, :Psi][1:n] - (real.(Ψ))[1:n]), label="Maya", title="Ψ")
    display(p)
end

# Comparing ϕ

begin
    n = 120
    p = plot(ηs[1:n], Maya_results_df[!, :Phi][1:n], label="Maya", title="ϕ", xlabel="η")
    plot!(ηs[1:n], ϕ[1:n] .|> real, label="Ben")
    display(p)
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/phi_comparison_varying_eta_constant_k.png")
end


# Saving the data

d = begin
    d = DataFrame(:η => ηs, :Txx => Txx, :Tyy => Tyy, :Tzz => Tzz, :Txz => Txz, :ψ => Ψ)
    CSV.write("scripts/benchmarks/comparison_to_Maya/gravitational_potential/numeric_varying_eta_constant_k_data.csv", d, delim=';', header=true)
end