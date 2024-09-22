module TemperatureFluctuations
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using FastSphericalHarmonics
using SphericalHarmonicModes
using EnvelopeApproximation.GravitationalPotentials
using QuadGK

function ISW_source(ks:: Vector{Vec3}, 
                    snapshot:: BubblesSnapShot, 
                    η_PT:: Float64, 
                    n_ϕ:: Int64, 
                    n_μ:: Int64,
                    ΔV:: Float64 = 1., 
                    a:: Float64 = 1., 
                    G:: Float64 = 1.; 
                    kwargs...)
    Ψ = quad_Ψ(ks,
               snapshot,
               η_PT,
               n_ϕ,
               n_μ,
               ΔV,
               a,
               G; kwargs...)
    ŋ = Ŋ(ks, current_bubbles(snapshot, η_PT), n_ϕ, n_μ, ΔV, a, G)
    return @. ŋ - 2 * Ψ
end

function spherical_ks(k:: Float64, n:: Int):: Vector{Vec3}
    Θ, Φ = sph_points(n)
    return [k * Vec3(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)) 
            for ϕ ∈ Φ, θ ∈ Θ]
end

function ISW_source_lm(k:: Float64, 
                       n:: Int,
                       snapshot:: BubblesSnapShot, 
                       η_PT:: Float64, 
                       n_ϕ:: Int64, 
                       n_μ:: Int64,
                       ΔV:: Float64 = 1., 
                       a:: Float64 = 1., 
                       G:: Float64 = 1.; 
                       kwargs...):: Matrix{ComplexF64}
    ks = spherical_ks(k, n)
    s = reshape(ISW_source(ks, snapshot, η_PT, n_ϕ, n_μ, ΔV, a, G), n, 2 * n - 1)
    return sph_transform!(s)
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function ISW_radial_integrand(k̃:: Float64, 
                              n:: Int,
                              snapshot:: BubblesSnapShot, 
                              η_PT:: Float64, 
                              χ_PT:: Float64,
                              n_ϕ:: Int64, 
                              n_μ:: Int64,
                              ΔV:: Float64 = 1., 
                              a:: Float64 = 1., 
                              G:: Float64 = 1.; 
                              kwargs...):: Matrix{ComplexF64}
    S = ISW_source_lm(k̃ / χ_PT, n, snapshot, η_PT, n_ϕ, n_μ, ΔV, a, G; kwargs...)
    l = 0:sph_lmax(n)
    cl = @. (cispi(l / 2) / (2 * π^2 * χ_PT ^ 3)) * sphericalbesselj(l, k̃) * k̃ ^ 2
    for (l̃, m) ∈ LM(l)
        S[sph_mode(l̃, m)] *= cl[l̃ + 1]
    end
    return S
end

function ISW(k:: Float64, 
             n:: Int,
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
    return quadgk(f, 0., Inf64)[1]
end

end