"""
    StressTensor

Ray-casting method for computing T_ij using deterministic spherical quadrature
and analytic time integration via the I₃(α; a, b) formula.

Key features:
- No dome-ring topology complexity
- No time quadrature (all integrals are closed-form)
- Pluggable spherical quadrature scheme interface
- Direct collision detection along rays

# References
The method discretizes the angular integral using spherical quadrature:
  ∫ dΩ f(n̂) ≈ ∑_a w_a f(n̂_a)

For each ray marker n̂_a, collision times are computed and the time integral
is evaluated analytically:
  A_ij^±(k) = (ΔV/3) ∑_a w_a n̂_{a,i} n̂_{a,j} e^{-ikz_i} e^{±ikt_i} v³ I₃(α_±; τ_start, τ_stop)

where I₃ is the closed-form integral of τ³ exp(iατ) from a to b.
"""
module StressTensor

import EnvelopeApproximation:
    TemporalWeight, CosineWeight, ConstantWeight, ComplexExponential,
    Kernel, allocate_accumulant, prepare_kernel!, accumulate_ray!
using EnvelopeApproximation.BubblesEvolution: Nucleation
using StaticArrays
using LinearAlgebra

using ..RayCastingEnvelopeIntegration:
    EnvelopeSource, SphericalQuadratureScheme, UniformSphericalCapScheme

abstract type StressTensorTerm end
abstract type Accumulant{W<:TemporalWeight} end

struct KineticTerm   <: StressTensorTerm end
struct PotentialTerm <: StressTensorTerm end
struct TotalStressTerm <: StressTensorTerm end

struct CosineAccumulant <: Accumulant{CosineWeight}
    A_plus::Matrix{ComplexF64}
    A_minus::Matrix{ComplexF64}
end

struct ConstantAccumulant <: Accumulant{ConstantWeight}
    A::Matrix{ComplexF64}
end

struct ComplexExponentialAccumulant{N} <: Accumulant{ComplexExponential{N}}
    A::Array{ComplexF64,3}
end

include("StressTensor/I2Kernels.jl")
include("StressTensor/I3Kernels.jl")
include("StressTensor/ModeAccumulation.jl")
include("StressTensor/TimeResolvedAccumulation.jl")


# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════
export SphericalQuadratureScheme,
       UniformSphericalCapScheme,
       StressTensorTerm,
       KineticTerm,
       PotentialTerm,
       TotalStressTerm,
       TemporalWeight,
       CosineWeight,
       ConstantWeight,
       ComplexExponential,
       Accumulant,
       CosineAccumulant,
       ConstantAccumulant,
       ComplexExponentialAccumulant,
       TimeResolvedStressTensorAccumulant,
       ModeWorkspace,
       CosineModeWorkspace,
       ConstantModeWorkspace,
       ComplexExponentialModeWorkspace,
       FourierStressTensorKernel,
       TimeResolvedStressTensorKernel,
       amplitudes

# ═══════════════════════════════════════════════════════════════════════════════
# Accumulant helpers
# ═══════════════════════════════════════════════════════════════════════════════

amplitudes(acc::CosineAccumulant) = (acc.A_plus, acc.A_minus)
amplitudes(acc::ConstantAccumulant) = acc.A
amplitudes(acc::ComplexExponentialAccumulant) = acc.A

check_ks(::KineticTerm, ks) = nothing
function check_ks(::Union{PotentialTerm, TotalStressTerm}, ks)
    any(iszero, ks) && throw(ArgumentError("k = 0 is not supported for PotentialTerm/TotalStressTerm"))
end
check_ks(terms::Tuple, ks) = foreach(t -> check_ks(t, ks), terms)

# Internal allocators used by the early-return path and by allocate_accumulant(kernel).
_alloc_accumulant(::CosineWeight, ::StressTensorTerm, Nk::Int) =
    CosineAccumulant(zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk))

_alloc_accumulant(::ConstantWeight, ::StressTensorTerm, Nk::Int) =
    ConstantAccumulant(zeros(ComplexF64, 6, Nk))

_alloc_accumulant(::ComplexExponential{N}, ::StressTensorTerm, Nk::Int) where {N} =
    ComplexExponentialAccumulant{N}(zeros(ComplexF64, 6, Nk, N))

# ═══════════════════════════════════════════════════════════════════════════════
# Fourier stress-tensor kernel
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FourierStressTensorKernel(term, weight, ks; ΔV=1.0, v=1.0)

Kernel that accumulates Fourier-mode stress-tensor amplitudes inside the generic
`envelope_integral` engine.
"""
struct FourierStressTensorKernel{T<:StressTensorTerm, W<:TemporalWeight, K<:AbstractVector{<:Real}} <: Kernel
    term::T
    weight::W
    ks::K
    mode_ws::ModeWorkspace{W}
    ΔV::Float64
    v::Float64
end

function FourierStressTensorKernel(
    term::T,
    weight::W,
    ks::K;
    ΔV::Real=1.0,
    v::Real=1.0,
) where {
    T<:StressTensorTerm,
    W<:TemporalWeight,
    K<:AbstractVector{<:Real},
}
    check_ks(term, ks)
    mode_ws = ModeWorkspace(weight, length(ks))
    return FourierStressTensorKernel(
        term, weight, ks, mode_ws, Float64(ΔV), Float64(v),
    )
end

# --- extensions of the RayCastingEnvelopeIntegration generic kernel interface ---

function allocate_accumulant(kernel::FourierStressTensorKernel)
    return _alloc_accumulant(kernel.weight, kernel.term, length(kernel.ks))
end

function prepare_kernel!(kernel::FourierStressTensorKernel, source::EnvelopeSource)
    prepare_source_modes!(kernel.mode_ws, kernel.weight, kernel.ks, source)
    return nothing
end

function accumulate_ray!(
    acc::Accumulant,
    kernel::FourierStressTensorKernel,
    ::EnvelopeSource,
    n̂::SVector{3,Float64},
    wΩ::Float64,
    τ_stop::Float64,
)
    accumulate_marker_modes!(
        acc, kernel.weight, kernel.term, kernel.ks,
        τ_stop, n̂, wΩ, kernel.mode_ws, kernel.ΔV, kernel.v,
    )
    return nothing
end

end # module StressTensor
