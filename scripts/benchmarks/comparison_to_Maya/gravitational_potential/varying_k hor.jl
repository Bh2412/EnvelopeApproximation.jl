using EnvelopeApproximation.GravitationalPotentials
using CSV, DataFrames
using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GravitationalPotentials: ψ_source, Ŋ, Φ, ψ
using Plots
using BenchmarkTools

# Loading Maya results
Maya_results_df = CSV.read("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Set6.csv", DataFrame)
ηs = Maya_results_df[!, :eta] |> unique
ks = Maya_results_df[!, :k_par] |> unique
begin
    Maya_Txx = Maya_results_df[!, :T_xx] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
    Maya_Tyy = Maya_results_df[!, :T_yy] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
    Maya_Tzz = Maya_results_df[!, :T_zz] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
    Maya_Txz = Maya_results_df[!, :T_zx] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
    Maya_ψ = Maya_results_df[!, :Psi] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
    Maya_Φ = Maya_results_df[!, :Phi] |> x -> reshape(x, length(ηs), length(ks)) |> permutedims
end

# Ssetting up the parameters
d = 1.
k_parallel = 0.
ΔV = 1.
kvecs = (x -> Vec3(x, 0., 0.)).(ks)
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
η_max = ηs[end]
snapshot = BubblesSnapShot(nucleations, η_max)

# computing Txx, Tyy, Tzz


function Tij(η:: Float64):: Array{ComplexF64, 2}
    return T_ij(kvecs, current_bubbles(snapshot, η); ΔV=ΔV, rtol=1e-3)
end

function ψsource(η:: Float64)
    return ψ_source(kvecs, snapshot, η; ΔV=ΔV, rtol=1e-3)
end

@time Ts = Tij.(ηs) |> x -> cat(x..., dims=3)

@time begin
    Txx = Ts[:, 1, :]
    Tyy = Ts[:, 4, :]
    Tzz = Ts[:, 6, :]
    Txy = Ts[:, 2, :]
    Txz = Ts[:, 3, :]
    Tyz = Ts[:, 5, :]        
    ψs = cat(ψsource.(ηs)..., dims=2)
end

# comparison Plots For Tij

begin
    p = plot(ks, Maya_Txx[:, end], label="Maya", title="Txx")
    plot!(ks, Txx[:, end] .|> real, label="Ben")     
    xlabel!("k_horizontal")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txx_comparison_varying_k_hor.png")
    display(p)
end

begin
    p = plot(ks, Maya_Tyy[:, end], label="Maya", title="Tyy")
    plot!(ks, Tyy[:, end] .|> real, label="Ben")     
    xlabel!("k_horizontal")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tyy_comparison_varying_k_hor.png")
    display(p)
end

begin
    p = plot(ks, Maya_Tzz[:, end], label="Maya", title="Tzz")
    plot!(ks, Tzz[:, end] .|> real, label="Ben")     
    xlabel!("k_horizontal")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Tzz_comparison_varying_k_hor.png")
    display(p)
end

begin
    p = plot(ks, Maya_Txz[:, end], label="Maya", title="Txz")
    plot!(ks, Txz[:, end] .|> real, label="Ben")     
    xlabel!("k_horizontal")
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txz_comparison_varying_k_hor.png")
    display(p)
end

begin
    p = plot(ks, ψs[:, end] .|> real, title="ψ source")    
    #savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/Txx_comparison_varying_eta_constant_k.png")
    display(p)
end

# comparing ψ

a, G = 1., 1.

function Ψ(η:: Float64):: Vector{ComplexF64}
    return ψ(kvecs, snapshot, η; ΔV=ΔV, a=a, G=G, rtol=1e-2)
end

@time _Ψ = ψ(kvecs, snapshot, η_max; rtol=1e-3)

_Ψ

@time ŋ = Ŋ(kvecs, current_bubbles(snapshot, η_max), a=a, G=G, rtol=1e-14)

ϕ = Φ(ŋ, _Ψ)

# plot

begin
    n = 119
    p = plot(ks[1:n], Maya_ψ[:, end][1:n], label="Maya", title="Ψ", xlabel="k_horizontal")
    plot!(ks[1:n], _Ψ[1:n] .|> real, label="Ben")
    display(p)
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/psi_comparison_varying_k_hor.png")
end

# difference plot

begin
    n = 60
    p = plot(ηs[1:n], (Maya_results_df[!, :Psi][1:n] - (real.(_Ψ))[1:n]), label="Maya", title="Ψ")
    display(p)
end

# Comparing ϕ

begin
    n = 119
    p = plot(ks[1:n], Maya_Φ[:, end][1:n], label="Maya", title="Φ", xlabel="k_horizontal")
    plot!(ks[1:n], ϕ[1:n] .|> real, label="Ben")
    display(p)
    savefig("scripts/benchmarks/comparison_to_Maya/gravitational_potential/phi_comparison_varying_k_hor.png")
end



# Saving the data

d = begin
    d = DataFrame(:η => ηs, :Txx => Txx, :Tyy => Tyy, :Tzz => Tzz, :Txz => Txz, :ψ => Ψ)
    CSV.write("scripts/benchmarks/comparison_to_Maya/gravitational_potential/numeric_varying_eta_constant_k_data.csv", d, delim=';', header=true)
end