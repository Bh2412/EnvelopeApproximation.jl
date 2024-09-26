module TemperatureFluctuations
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using FastSphericalHarmonics
using SphericalHarmonicModes
using SpecialFunctions
using EnvelopeApproximation.GravitationalPotentials
import EnvelopeApproximation.GravitationalPotentials: ψ_source
using QuadGK
using HCubature
using StaticArrays

function ISW_ψ_source(τ:: Float64, 
                      ks:: Vector{Vec3}, 
                      snapshot:: BubblesSnapShot, 
                      η_PT:: Float64, 
                      n_ϕ:: Int64, 
                      n_μ:: Int64,
                      ΔV:: Float64 = 1., 
                      a:: Float64 = 1., 
                      G:: Float64 = 1.; 
                      kwargs...)
    # This assumes the ssnapshot is given at the end of the PT
    Ψs = ψ_source(ks,
                  snapshot,
                  η_PT,
                  n_ϕ,
                  n_μ,
                  ΔV,
                  a,
                  G; kwargs...)
    snapshot.t
    return @. ŋ - 2 * Ψ
end

function spherical_ks(k:: Float64, n:: Int):: Matrix{Vec3}
    Θ, Φ = sph_points(n)
    return [k * Vec3(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)) 
            for  θ ∈ Θ, ϕ ∈ Φ]
end

function ψlm_source(k:: Float64, 
                    η:: Float64,
                    n:: Int,
                    snapshot:: BubblesSnapShot, 
                    n_ϕ:: Int64, 
                    n_μ:: Int64,
                    ΔV:: Float64 = 1., 
                    a:: Float64 = 1., 
                    G:: Float64 = 1.; 
                    kwargs...):: Matrix{ComplexF64}
    ks = reshape(spherical_ks(k, n), :)
    η_end = snapshot.t
    s = reshape(ψ_source(ks, snapshot, η_end, n_ϕ, n_μ, ΔV, a, G), n, 2 * n - 1)
    return sph_transform!(s)
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function ψ_contribution_integrand(κ:: Float64,
                                  η:: Float64,
                                  n:: Int,
                                  snapshot:: BubblesSnapShot, 
                                  χ_PT:: Float64,
                                  n_ϕ:: Int64, 
                                  n_μ:: Int64,
                                  ΔV:: Float64 = 1., 
                                  a:: Float64 = 1., 
                                  G:: Float64 = 1.; 
                                  kwargs...):: Matrix{ComplexF64}
    S = ψlm_source(κ / χ_PT, η, n, snapshot, n_ϕ, n_μ, ΔV, a, G; kwargs...)
    l = 0:sph_lmax(n)
    cl = @. (cispi(l / 2) / (2 * π^2 * χ_PT ^ 3)) * sphericalbesselj(l, κ) * κ ^ 2
    for (l̃, m) ∈ LM(l)
        S[sph_mode(l̃, m)] *= cl[l̃ + 1]
    end
    return S
end

function ψ_contribution(n:: Int,
                        snapshot:: BubblesSnapShot, 
                        χ_PT:: Float64,
                        n_ϕ:: Int64, 
                        n_μ:: Int64,
                        ΔV:: Float64 = 1., 
                        a:: Float64 = 1., 
                        G:: Float64 = 1.; 
                        kwargs...):: Matrix{ComplexF64}
    f̃(κ:: Float64, η::Float64):: Matrix{ComplexF64} = ψ_contribution_integrand(κ, η,
                                                                               n, 
                                                                               snapshot, 
                                                                               χ_PT, n_ϕ, 
                                                                               n_μ, ΔV, 
                                                                               a, G; 
                                                                               kwargs...)
    η_end = snapshot.t
    f̃_η(η:: Float64):: Matrix{ComplexF64} = quadgk(κ -> f̃(κ, η), 0., Inf; kwargs...) * (η_end - η)
    return quadgk(f̃_η, 0., η_end; kwargs...)
end

export ISW

function ISW(n:: Int,
             snapshot:: BubblesSnapShot, 
             η_PT:: Float64, 
             χ_PT:: Float64,
             n_ϕ:: Int64, 
             n_μ:: Int64,
             ΔV:: Float64 = 1., 
             a:: Float64 = 1., 
             G:: Float64 = 1.; 
             kwargs...):: Matrix{ComplexF64}
    f(k̃:: Float64):: Matrix{ComplexF64} = ISW_radial_integrand(k̃, n, snapshot, 
                                                               η_PT, χ_PT,
                                                               n_ϕ,
                                                               n_μ,
                                                               ΔV,
                                                               a, G; kwargs...)
    return quadgk(f, 0., Inf64; kwargs...)[1]
end

end