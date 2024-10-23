module TemperatureFluctuations
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
using FastSphericalHarmonics
using SphericalHarmonicModes
using StaticArrays
using EnvelopeApproximation.GravitationalPotentials
using SpecialFunctions
using QuadGK

function spherical_unit_vectors(n:: Int):: Matrix{Vec3}
    Θ, Φ = sph_points(n)
    return [Vec3(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)) 
            for θ ∈ Θ, ϕ ∈ Φ]
end

function k̂ik̂jTij_lm(k:: Float64, n̂s:: Matrix{Vec3}, bubbles:: Bubbles, 
                    arcs:: Dict{Int64, Vector{IntersectionArc}},
                    n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
                    ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    V:: Matrix{ComplexF64} = similar(n̂s, ComplexF64)
    @inbounds for (i, n̂, n̂_rotation) in zip(eachindex(n̂s), n̂s, n̂_rotations)
        V[i] = k̂ik̂jTij(k * n̂, bubbles, arcs, n̂_rotation, ΔV; kwargs...)
    end
    return sph_transform!(V)
end

function ŋ_lm(k:: Float64, n̂s:: Matrix{Vec3}, bubbles:: Bubbles, 
              arcs:: Dict{Int64, Vector{IntersectionArc}},
              n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
              ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
              V:: Matrix{ComplexF64} = similar(n̂s, ComplexF64)
    @inbounds for (i, n̂, n̂_rotation) in zip(eachindex(n̂s), n̂s, n̂_rotations)
        V[i] = ŋ_source(k * n̂, bubbles, arcs, n̂_rotation, ΔV; kwargs...) / (k^2)
    end
    return sph_transform!(V)
end

sphericalbesselj(ν, x) = sqrt(π / (2 * x)) * besselj(ν + 1/2, x)

function ψ_contribution_integrand(κ:: Float64, η:: Float64, χ_PT:: Float64, η_PT,
                                  n̂s:: Matrix{Vec3}, snapshot:: BubblesSnapShot, 
                                  n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
                                  ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    n, _ = size(n̂s)
    ls = 0:sph_lmax(n)
    c = (η_PT - η) * κ ^ 2
    cl = @. sphericalbesselj(ls, κ) * c * cispi(ls / 2)
    bubbles = current_bubbles(snapshot, η)
    arcs = intersection_arcs(bubbles)
    _k̂ik̂jTij_lm = k̂ik̂jTij_lm(κ / χ_PT, n̂s, bubbles, 
                             arcs, n̂_rotations, ΔV; kwargs...)
    @inbounds for (l, m) ∈ LM(ls)
        _k̂ik̂jTij_lm[sph_mode(l, m)] *= cl[l + 1]
    end
    return _k̂ik̂jTij_lm
end

function ŋ_contribution_integrand(κ:: Float64, χ_PT:: Float64,
                                  n̂s:: Matrix{Vec3}, bubbles:: Bubbles,
                                  arcs:: Dict{Int64, Vector{IntersectionArc}},
                                  n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
                                  ΔV:: Float64 = 1.; kwargs...)
    n, _ = size(n̂s)
    ls = 0:sph_lmax(n)
    cl = @. sphericalbesselj(ls, κ) * (κ ^ 2) * cispi(ls / 2)
    _ŋ_source_lm = ŋ_lm(κ / χ_PT, n̂s, bubbles, 
                        arcs, n̂_rotations, ΔV; kwargs...)
    @inbounds for (l, m) ∈ LM(ls)
        _ŋ_source_lm[sph_mode(l, m)] *= cl[l + 1]
    end
    return _ŋ_source_lm
end

function ψ_contribution(χ_PT:: Float64, η_PT:: Float64,
                        n̂s:: Matrix{Vec3}, snapshot:: BubblesSnapShot,
                        n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
                        ΔV:: Float64 = 1., a:: Float64 = 1., 
                        G:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    integrand(κ:: Float64, η:: Float64):: Matrix{ComplexF64} = ψ_contribution_integrand(κ, η, χ_PT, η_PT, n̂s, snapshot, n̂_rotations, 
                                                                                          ΔV; kwargs...)                                                                                      
    η_integrand(η:: Float64) :: Matrix{ComplexF64} = quadgk(κ -> integrand(κ, η), 0., Inf64; kwargs...)[1]
    V = quadgk(η_integrand, 0., η_PT; kwargs...)[1]
    return (-4G * a^2 / (π * χ_PT ^ 3)) .* V
end

function ŋ_contribution(χ_PT:: Float64, η_PT:: Float64,
                        n̂s:: Matrix{Vec3}, snapshot:: BubblesSnapShot,
                        n̂_rotations:: Matrix{<: SMatrix{3, 3, Float64}}, 
                        ΔV:: Float64 = 1.; kwargs...):: Matrix{ComplexF64}
    bubbles = current_bubbles(snapshot, η_PT)
    arcs = intersection_arcs(bubbles)
    integrand(κ:: Float64):: Matrix{ComplexF64} = ŋ_contribution_integrand(κ, χ_PT, n̂s, bubbles, arcs, n̂_rotations, ΔV; kwargs...)
    return (1 / (2 * (π ^ 2) * χ_PT ^ 3)) * quadgk(integrand, 0., Inf64; kwargs...)[1]
end

function ISW(snapshot:: BubblesSnapShot, 
             χ_PT:: Float64,
             η_PT:: Float64, 
             lmax:: Int64;
             ΔV:: Float64 = 1., 
             a:: Float64 = 1., 
             G:: Float64 = 1.,
             kwargs...):: Matrix{ComplexF64}
    n̂s = spherical_unit_vectors(lmax + 1)
    n̂_rotations = align_ẑ.(n̂s)
    _ŋ_contribution = ŋ_contribution(χ_PT, η_PT, n̂s, snapshot, n̂_rotations, ΔV; kwargs...)
    _ψ_contribution = ψ_contribution(χ_PT, η_PT, n̂s, snapshot, n̂_rotations, ΔV, a, G; kwargs...)
    return _ŋ_contribution + _ψ_contribution
end

end