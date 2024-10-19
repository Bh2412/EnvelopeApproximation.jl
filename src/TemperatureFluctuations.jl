module TemperatureFluctuations
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using FastSphericalHarmonics
using SphericalHarmonicModes
using EnvelopeApproximation.GravitationalPotentials
using SpecialFunctions
using QuadGK

function ISW_source(ks:: Vector{Vec3}, 
                    snapshot:: BubblesSnapShot, 
                    η_PT:: Float64;
                    ΔV:: Float64 = 1., 
                    a:: Float64 = 1., 
                    G:: Float64 = 1.,
                    kwargs...)
    Ψ = ψ(ks,
          snapshot,
          η_PT;
          ΔV=ΔV,
          a=a,
          G=G, kwargs...)
    ŋ = Ŋ(ks, current_bubbles(snapshot, η_PT), ΔV=ΔV, a=a, G=G)
    return @. ŋ - 2 * Ψ
end

function spherical_unit_vectors(n:: Int):: Vector{Vec3}
    Θ, Φ = sph_points(n)
    return [Vec3(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)) 
            for ϕ ∈ Φ, θ ∈ Θ][:]
end

function T_k(ks:: Vector)
end

function ISW_source_lm(k:: Float64, 
                       snapshot:: BubblesSnapShot, 
                       η_PT:: Float64,
                       n:: Int;
                       ΔV:: Float64 = 1., 
                       a:: Float64 = 1., 
                       G:: Float64 = 1.,
                       kwargs...):: Matrix{ComplexF64}
    ks = spherical_unit_vectors(k, n)
    s = reshape(ISW_source(ks, snapshot, η_PT; ΔV=ΔV, a=a, G=G), n, 2 * n - 1)
    return sph_transform!(s)
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function ISW_radial_integrand(k̃:: Float64, 
                              snapshot:: BubblesSnapShot, 
                              η_PT:: Float64, 
                              n:: Int,
                              χ_PT:: Float64;
                              ΔV:: Float64 = 1., 
                              a:: Float64 = 1., 
                              G:: Float64 = 1., 
                              kwargs...):: Matrix{ComplexF64}
    S = ISW_source_lm(k̃ / χ_PT, snapshot, η_PT, n; ΔV=ΔV, a=a, G=G, kwargs...)
    l = 0:sph_lmax(n)
    cl = @. (cispi(l / 2) / (2 * π^2 * χ_PT ^ 3)) * sphericalbesselj(l, k̃) * k̃ ^ 2
    for (l̃, m) ∈ LM(l)
        S[sph_mode(l̃, m)] *= cl[l̃ + 1]
    end
    return S
end

function ISW(snapshot:: BubblesSnapShot, 
             η_PT:: Float64,
             n:: Int, 
             χ_PT:: Float64;
             ΔV:: Float64 = 1., 
             a:: Float64 = 1., 
             G:: Float64 = 1.,
             kwargs...):: Matrix{ComplexF64}
    f(k̃:: Float64):: Matrix{ComplexF64} = ISW_radial_integrand(k̃, snapshot, 
                                                               η_PT, n, χ_PT;
                                                               ΔV,
                                                               a, G, kwargs...)
    return quadgk(f, 0., Inf64)[1]
end

end