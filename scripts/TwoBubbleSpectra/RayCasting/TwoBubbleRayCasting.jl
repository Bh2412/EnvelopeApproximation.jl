using StaticArrays
using QuadGK

import EnvelopeApproximation.RayCastingLightConesSurface:
    allocate_accumulant, prepare_kernel!, accumulate_ray!
using EnvelopeApproximation.RayCastingLightConesSurface:
    LightConeSource, lightcone_sources, build_lightcone_context,
    integrate_lightcone_surfaces, SphericalQuadratureScheme,
    UniformSphericalCapScheme, get_markers
using EnvelopeApproximation.RayCastingStressEnergyTensor:
    StressTensorTerm, KineticTerm, PotentialTerm, TotalStressTerm,
    TemporalWeight, ConstantWeight, ConstantAccumulant, ConstantModeWorkspace,
    ModeWorkspace, amplitudes
import EnvelopeApproximation.RayCastingStressEnergyTensor: prepare_source_modes!, I2_zero_lower, I3_zero_lower
using EnvelopeApproximation.BubbleBasics: Point3
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Vacuum
using EnvelopeApproximation.EnvelopeAnalysis: align_ẑ, symmetric_tensor_inverse_rotation
using EnvelopeApproximation.StressEnergyTensorContractions: contract

include(joinpath(@__DIR__, "..", "TwoPointStressEnergyTensor.jl"))

# ============================================================
# Local accumulation helpers
# ============================================================

@inline _outer_prod(n̂::SVector{3,Float64}) = SVector{6,Float64}(
    n̂[1]*n̂[1], n̂[1]*n̂[2], n̂[1]*n̂[3],
    n̂[2]*n̂[2], n̂[2]*n̂[3], n̂[3]*n̂[3])

@inline function _accumulate_tensor!(A::Matrix{ComplexF64}, q::Int,
                                     amp::ComplexF64, nn::SVector{6,Float64})
    @inbounds for I in 1:6
        A[I, q] = muladd(amp, nn[I], A[I, q])
    end
end

@inline function _accumulate_diagonal!(A::Matrix{ComplexF64}, q::Int, amp::ComplexF64)
    @inbounds A[1, q] += amp; A[4, q] += amp; A[6, q] += amp
end

# ============================================================
# Split integrator: exact on [0, τc], numerical on [τc, τ_stop]
# ============================================================

@inline function J2J3_gaussian_late(
    α::Float64,
    t_i::Float64,
    τ_stop::Float64,
    cutoff::GaussianLateTimeCutoff,
    τ_scheme,
)::SVector{2,ComplexF64}
    τ_stop <= 0.0 && return SVector{2,ComplexF64}(0.0 + 0.0im, 0.0 + 0.0im)

    τc = cutoff.tc - t_i

    if τ_stop <= τc
        # entire ray sits inside the flat C = 1 region — fully analytic
        return SVector{2,ComplexF64}(I2_zero_lower(α, τ_stop), I3_zero_lower(α, τ_stop))
    end

    if τc <= 0.0
        # Gaussian already active at τ = 0 — full numerical integration
        return integrate(τ_scheme, function(τ)
            τ2  = τ * τ
            fac = cutoff(t_i + τ) * cis(α * τ)
            SVector{2,ComplexF64}(fac * τ2, fac * τ2 * τ)
        end, 0.0, τ_stop)
    end

    # Mixed: analytic on [0, τc] (C = 1), smooth Gaussian tail on [τc, τ_stop]
    early = SVector{2,ComplexF64}(I2_zero_lower(α, τc), I3_zero_lower(α, τc))

    late = integrate(τ_scheme, function(τ)
        τ2  = τ * τ
        fac = cutoff(t_i + τ) * cis(α * τ)
        SVector{2,ComplexF64}(fac * τ2, fac * τ2 * τ)
    end, τc, τ_stop)

    return early + late
end

# ============================================================
# Kernel struct
# ============================================================

struct GaussianCutoffFourierStressTensorKernel{
    T  <: StressTensorTerm,
    W  <: TemporalWeight,
    KS <: AbstractVector{<:Real},
    MW <: ModeWorkspace{W},
    C,
    Q,
}
    term::T
    weight::W
    ks::KS
    mode_ws::MW
    cutoff::C
    τ_scheme::Q
    ΔV::Float64
    v::Float64
end

# ============================================================
# RayCastingLightConesSurface interface
# ============================================================

function allocate_accumulant(kernel::GaussianCutoffFourierStressTensorKernel{<:StressTensorTerm, ConstantWeight})
    return ConstantAccumulant(zeros(ComplexF64, 6, length(kernel.ks)))
end

function prepare_kernel!(
    kernel::GaussianCutoffFourierStressTensorKernel{<:StressTensorTerm, ConstantWeight},
    source::LightConeSource,
)
    prepare_source_modes!(kernel.mode_ws, kernel.weight, kernel.ks, source)
    return nothing
end

function accumulate_ray!(
    acc::ConstantAccumulant,
    kernel::GaussianCutoffFourierStressTensorKernel{<:StressTensorTerm, ConstantWeight},
    source::LightConeSource,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    τ_stop::Float64,
)
    accumulate_marker_modes_gaussian_cutoff!(
        acc, kernel.weight, kernel.term, kernel.ks,
        τ_stop, source.time, n̂, wΩ, kernel.mode_ws,
        kernel.cutoff, kernel.τ_scheme, kernel.ΔV, kernel.v,
    )
    return nothing
end

# ============================================================
# Numerical integration — ConstantWeight only
# ============================================================

function accumulate_marker_modes_gaussian_cutoff!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::KineticTerm,
    ks,
    τ_stop::Float64,
    t_i::Float64,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    ws::ConstantModeWorkspace,
    cutoff::GaussianLateTimeCutoff,
    τ_scheme,
    ΔV::Float64,
    v::Float64,
)
    nᵢnⱼ = _outer_prod(n̂)
    n3 = n̂[3]
    prefactor = wΩ * (ΔV / 3.0) * v^3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        α = -k * v * n3
        I3 = J2J3_gaussian_late(α, t_i, τ_stop, cutoff, τ_scheme)[2]
        _accumulate_tensor!(acc.A, q, prefactor * ws.phase[q] * I3, nᵢnⱼ)
    end
    return nothing
end

function accumulate_marker_modes_gaussian_cutoff!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::PotentialTerm,
    ks,
    τ_stop::Float64,
    t_i::Float64,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    ws::ConstantModeWorkspace,
    cutoff::GaussianLateTimeCutoff,
    τ_scheme,
    ΔV::Float64,
    v::Float64,
)
    n3 = n̂[3]
    prefactor = wΩ * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        iszero(k) && throw(ArgumentError("PotentialTerm with k = 0 is not supported"))
        α = -k * v * n3
        I2 = J2J3_gaussian_late(α, t_i, τ_stop, cutoff, τ_scheme)[1]
        _accumulate_diagonal!(acc.A, q, (prefactor / k) * ws.phase[q] * I2)
    end
    return nothing
end

function accumulate_marker_modes_gaussian_cutoff!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::TotalStressTerm,
    ks,
    τ_stop::Float64,
    t_i::Float64,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    ws::ConstantModeWorkspace,
    cutoff::GaussianLateTimeCutoff,
    τ_scheme,
    ΔV::Float64,
    v::Float64,
)
    τ_stop <= 0.0 && return nothing

    nn  = _outer_prod(n̂)
    n3  = n̂[3]
    kinetic_pf   = wΩ * (ΔV / 3.0) * v^3
    potential_pf = wΩ * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        iszero(k) && throw(ArgumentError("k = 0 is not supported (PotentialTerm diverges)"))

        α = -k * v * n3
        I2, I3 = J2J3_gaussian_late(α, t_i, τ_stop, cutoff, τ_scheme)

        ph = ws.phase[q]
        _accumulate_tensor!(acc.A, q, kinetic_pf * ph * I3, nn)
        _accumulate_diagonal!(acc.A, q, (potential_pf / k) * ph * I2)
    end

    return nothing
end

# ============================================================
# Two-bubble ray-casting amplitude — private helpers
# ============================================================

function _two_bubble_setup(d::Float64, v::Float64, τ_over_d::Float64)
    τ, cutoff = two_bubble_cutoff(d; τ_over_d = τ_over_d)
    nuc1     = Nucleation((time = 0.0, site = Point3(0.0, 0.0, -d / 2)))
    nuc2     = Nucleation((time = 0.0, site = Point3(0.0, 0.0,  d / 2)))
    snapshot = BubblesSnapShot([nuc1, nuc2], τ)
    space    = BoxSpace(2.0 * (v * τ + d), Point3(0.0, 0.0, 0.0))
    return τ, cutoff, snapshot, space
end

function _two_bubble_ray_cast(
    ks_f, snapshot, space, cutoff,
    quadrature::SphericalQuadratureScheme,
    τ_scheme, ρ_vac::Float64, v::Float64,
)::Matrix{ComplexF64}
    any(iszero, ks_f) && throw(ArgumentError("k = 0 is not supported (PotentialTerm diverges)"))
    context = build_lightcone_context(snapshot, space, Vacuum(); v = v)
    sources = lightcone_sources(snapshot)
    markers = get_markers(quadrature)
    mode_ws = ModeWorkspace(ConstantWeight(), length(ks_f))
    kernel  = GaussianCutoffFourierStressTensorKernel(
        TotalStressTerm(), ConstantWeight(), ks_f, mode_ws,
        cutoff, τ_scheme, ρ_vac, v,
    )
    (acc,) = integrate_lightcone_surfaces((kernel,), sources, context, markers;
                                          τ_min = 1.0e-12)
    return amplitudes(acc)
end

# ============================================================
# Two-bubble ray-casting amplitude — public API
# ============================================================

"""
    two_bubble_Tij_amplitude_ray_casting(ks, d; ...)

Ray-casting analogue of `two_bubble_Tij_amplitude` (ConstantWeight, no time phase).
Two bubbles at z = ±d/2, wavevector along ẑ. Returns 6×Nk Matrix{ComplexF64}.
"""
function two_bubble_Tij_amplitude_ray_casting(
    ks::AbstractVector{<:Real},
    d::Float64;
    ρ_vac::Float64                        = 1.0,
    τ_over_d::Float64                     = 1.2,
    τ_scheme::IntegrationScheme           = GaussLegendreScheme(96),
    quadrature::SphericalQuadratureScheme = UniformSphericalCapScheme(16, 32),
    v::Float64                            = 1.0,
)::Matrix{ComplexF64}
    _, cutoff, snapshot, space = _two_bubble_setup(d, v, τ_over_d)
    ks_f = ks isa AbstractRange ? ks : collect(Float64, ks)
    return _two_bubble_ray_cast(ks_f, snapshot, space, cutoff, quadrature, τ_scheme, ρ_vac, v)
end

"""
    two_bubble_Tij_amplitude_ray_casting(ks, k̂, d; ...)

As above but for an arbitrary wavevector direction k̂. Rotates the snapshot so k̂
aligns with ẑ, delegates to the base overload, then rotates T_ij back.
Returns 6×Nk Matrix{ComplexF64} in the original (unrotated) frame.
"""
function two_bubble_Tij_amplitude_ray_casting(
    ks::AbstractVector{<:Real},
    k̂::SVector{3,Float64},
    d::Float64;
    ρ_vac::Float64                        = 1.0,
    τ_over_d::Float64                     = 1.2,
    τ_scheme::IntegrationScheme           = GaussLegendreScheme(96),
    quadrature::SphericalQuadratureScheme = UniformSphericalCapScheme(16, 32),
    v::Float64                            = 1.0,
)::Matrix{ComplexF64}
    _, cutoff, base_snapshot, space = _two_bubble_setup(d, v, τ_over_d)
    R        = align_ẑ(k̂)
    snapshot = R * base_snapshot
    ks_f     = ks isa AbstractRange ? ks : collect(Float64, ks)
    A_rot    = _two_bubble_ray_cast(ks_f, snapshot, space, cutoff, quadrature, τ_scheme, ρ_vac, v)
    D        = symmetric_tensor_inverse_rotation(R)
    return Matrix{Float64}(D) * A_rot
end

# ============================================================
# Direction-averaged contraction — ray casting
# ============================================================

"""
    direction_average_contraction_ray_casting(ωs, d, projectors; ...)

Ray-casting analogue of `direction_average_contraction` (ConstantWeight, no time phase).
Returns Dict{Symbol, Vector{ComplexF64}}, one vector per projector, indexed by ω.
"""
function direction_average_contraction_ray_casting(
    ωs::AbstractVector{<:Real},
    d::Float64,
    projectors;
    ρ_vac::Float64                        = 1.0,
    τ_over_d::Float64                     = 1.2,
    τ_scheme::IntegrationScheme           = GaussLegendreScheme(96),
    quadrature::SphericalQuadratureScheme = UniformSphericalCapScheme(16, 32),
    μ_scheme::IntegrationScheme           = GaussLegendreScheme(128),
    v::Float64                            = 1.0,
)
    names = [name for (name, _) in projectors]
    Np    = length(projectors)
    ωs_f  = ωs isa AbstractRange ? ωs : collect(Float64, ωs)
    Nω    = length(ωs_f)

    function μ_integrand(μ::Float64)
        sinθ = sqrt(max(0.0, 1.0 - μ^2))
        k̂_μ  = SVector{3,Float64}(sinθ, 0.0, μ)

        A = two_bubble_Tij_amplitude_ray_casting(ωs_f, k̂_μ, d;
                ρ_vac=ρ_vac, τ_over_d=τ_over_d, τ_scheme=τ_scheme,
                quadrature=quadrature, v=v)   # 6 × Nω, already in original frame

        result = Matrix{ComplexF64}(undef, Np, Nω)
        for qi in 1:Nω
            Tij  = SVector{6,ComplexF64}(A[:, qi])
            Tmat = Tij * Tij'
            for (p_i, (_, Λ)) in enumerate(projectors)
                result[p_i, qi] = contract(Tmat, Λ(k̂_μ))
            end
        end
        return result
    end

    val = integrate(μ_scheme, μ_integrand, -1.0, 1.0)   # Np × Nω
    return Dict(names[i] => 2π .* val[i, :] for i in eachindex(names))
end
