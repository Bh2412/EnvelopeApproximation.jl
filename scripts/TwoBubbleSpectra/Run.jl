using EnvelopeApproximation.StressEnergyTensorContractions
using JLD2
using Dates
include(joinpath(@__DIR__, "TwoPointStressEnergyTensor.jl"))

d          = 1.0
projectors = (:TT => Λ_TT, :T => Λ_T, :pp => Λ_pp, :σσ => Λ_σσ, :pσ => Λ_pσ)
θ_scheme   = GaussLegendreScheme(128)
t_scheme   = GaussLegendreScheme(32)
μ_scheme   = GaussLegendreScheme(128)
Nω         = 1000
ωs         = collect(logrange(1e-2, 1e2, Nω))

data = Dict(name => Vector{ComplexF64}(undef, Nω) for (name, _) in projectors)

for (i, ω) in enumerate(ωs)
    println("ω = $ω  ($i / $Nω)")
    vals = direction_average_contraction(ω, d, projectors;
                                         include_time_phase=false,
                                         θ_scheme=θ_scheme,
                                         t_scheme=t_scheme,
                                         μ_scheme=μ_scheme)
    for (name, v) in vals
        data[name][i] = v
    end
end

timestamp   = Dates.format(now(), "yyyy_mm_dd_HH_MM")
output_path = joinpath(@__DIR__, "two_bubble_spectrum_$(timestamp).jld2")
jldsave(output_path; ωs, data)
println("Saved to $output_path")
