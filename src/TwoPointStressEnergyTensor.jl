module TwoPointStressEnergyTensorModule

import EnvelopeApproximation: TemporalWeight, CosineWeight, ConstantWeight
using EnvelopeApproximation.StressEnergyTensorComponents
using EnvelopeApproximation.StressTensor: RayCastingSphericalQuadrature,
    StressTensorTerm, KineticTerm, PotentialTerm,
    Accumulant, CosineAccumulant, ConstantAccumulant,
    amplitudes, ray_T_ij
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, current_bubbles
using EnvelopeApproximation.Spaces: AbstractSpace, BoxSpace, volume
using EnvelopeApproximation.BoundaryConditions: BoundaryCondition, Periodic
using QuadGK

export TwoPointStressEnergyTensor,
    TemporalWeight, CosineWeight, ConstantWeight,
    TimeIntegrationStrategy, QuadGKIntegration, UniformGridIntegration,
    TimeIntegratedTwoPointStressEnergyTensor,
    TimeIntegratedTwoPointStressEnergyTensorPieces,
    RayCastingCosineTimeIntegratedTwoPointStressTensor

abstract type TimeIntegrationStrategy end

struct QuadGKIntegration <: TimeIntegrationStrategy
    rtol::Float64
    atol::Float64
    maxevals::Int
end
QuadGKIntegration(; rtol::Real=1e-3, atol::Real=0.0, maxevals::Int=typemax(Int)) =
    QuadGKIntegration(Float64(rtol), Float64(atol), maxevals)

struct UniformGridIntegration <: TimeIntegrationStrategy
    N_t::Int
end

"""
    TwoPointStressEnergyTensor(t1, t2, ωs, snapshot, space, boundary_condition, strategy;
                               ΔV=1.0, bubble_indices=:)

Computes the two-point stress-energy tensor correlator ⟨T_{ij}(k,t1) T_{lm}(-k,t2)⟩
for k along ẑ, returning a 6×6×Nk array of ComplexF64.

Both tensor indices follow the symmetric tensor convention: xx=1, xy=2, xz=3, yy=4, yz=5, zz=6.
Element [ij, lm, ki] = T1[ij, ki] * conj(T2[lm, ki]) / V.

Pass `bubble_indices=1:3` to sum only over those original bubble indices. For periodic
boundary conditions, all periodic copies of the selected original bubbles are included
in the sum while the envelope domes are still computed against the full periodic system.
"""
function TwoPointStressEnergyTensor(t1::Real, t2::Real, ωs::AbstractVector{<:Real},
                                     snapshot::BubblesSnapShot, space::AbstractSpace,
                                     boundary_condition::BoundaryCondition,
                                     strategy::∂iϕ∂jϕ_AzimuthalReduction;
                                     ΔV::Float64=1.0,
                                     bubble_indices=:)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    bubbles1 = current_bubbles(snapshot, Float64(t1))
    bubbles2 = current_bubbles(snapshot, Float64(t2))
    T1 = ∂iϕ∂jϕ(ωs_f, bubbles1, space, boundary_condition, strategy.plan, strategy.buffer;
                 ΔV=ΔV, bubble_indices=bubble_indices)
    T2 = ∂iϕ∂jϕ(ωs_f, bubbles2, space, boundary_condition, strategy.plan, strategy.buffer;
                 ΔV=ΔV, bubble_indices=bubble_indices)
    V_inv = 1.0 / volume(space)
    return reshape(T1, 6, 1, :) .* conj.(reshape(T2, 1, 6, :)) .* V_inv
end

# ── ∂iϕ∂jϕ_AzimuthalReduction · CosineWeight ────────────────────────────────

"""
    QuadGkCosineTimeIntegratedTwoPointStressEnergyTensor(ωs, snapshot, space,
        boundary_condition, strategy; ΔV=1.0, bubble_indices=:,
        rtol=1e-3, atol=0.0, maxevals=typemax(Int))

Computes the time-integrated two-point correlator with weighting cos(k(t1−t2)):

  ∫∫ cos(k(t1−t2)) ⟨T_{ij}(k,t1) T_{lm}(-k,t2)⟩ dt1 dt2

using the factorisation cos(k Δt) = (e^{ikΔt} + e^{-ikΔt})/2, which reduces the 2D
integral to two 1D integrals:

  = (1/2V) [ A⁺_{ij}(k) conj(A⁺_{lm}(k)) + A⁻_{ij}(k) conj(A⁻_{lm}(k)) ]

where  A±_{ij}(k) = ∫ e^{±ikt} T_{ij}(k,t) dt.

Both integrals share a single QuadGK pass: T_{ij}(k,t) is evaluated once per quadrature
point and multiplied by e^{+ikt} and e^{-ikt} simultaneously.

Returns a 6×6×Nk array of ComplexF64.
"""
function QuadGkCosineTimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::∂iϕ∂jϕ_AzimuthalReduction;
    ΔV::Float64 = 1.0,
    bubble_indices = :,
    rtol::Real = 1e-3,
    atol::Real = 0.0,
    maxevals::Int = typemax(Int)
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    V_inv   = 1.0 / volume(space)

    # Single quadrature pass: returns [vec(e^{+ikt} T); vec(e^{-ikt} T)], length 12Nk
    function integrand(t::Float64)
        bubbles = current_bubbles(snapshot, t)
        T = ∂iϕ∂jϕ(ωs_f, bubbles, space, boundary_condition,
                     strategy.plan, strategy.buffer; ΔV=ΔV, bubble_indices=bubble_indices)
        phases = cis.(ωs_f .* t)                          # e^{+ikt} per k
        return vcat(vec(T .* reshape(phases,        1, Nk)),
                    vec(T .* reshape(conj.(phases), 1, Nk)))
    end

    combined, _ = quadgk(integrand, t_start, t_end;
                         rtol=rtol, atol=atol, maxevals=maxevals)

    A_plus  = reshape(combined[1:6Nk],     6, Nk)   # ∫ e^{+ikt} T_{ij}(k,t) dt
    A_minus = reshape(combined[6Nk+1:end], 6, Nk)   # ∫ e^{-ikt} T_{ij}(k,t) dt

    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        ap = A_plus[:,  ki]
        am = A_minus[:, ki]
        # ap * ap' computes ap[i]*conj(ap[j]) — the outer product A⁺_{ij} conj(A⁺_{lm})
        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

function UniformGridCosineIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::∂iϕ∂jϕ_AzimuthalReduction;
    N_t::Int,
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    Δt      = (t_end - t_start) / (N_t - 1)
    t_grid  = t_start .+ (0:N_t-1) .* Δt

    # Evaluate T_ij sequentially (leaving parallelization for the outer Monte Carlo loop)
    T_vals = Array{ComplexF64, 3}(undef, 6, Nk, N_t)
    for m in 1:N_t
        bubbles = current_bubbles(snapshot, Float64(t_grid[m]))
        T_vals[:, :, m] = ∂iϕ∂jϕ(ωs_f, bubbles, space, boundary_condition,
                                    strategy.plan, strategy.buffer; ΔV=ΔV,
                                    bubble_indices=bubble_indices)
    end

    # Trapezoidal weights: endpoints get half weight
    T_vals[:, :, 1]   .*= 0.5
    T_vals[:, :, end] .*= 0.5

    V_inv  = 1.0 / volume(space)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)

    # Directly compute the discrete integral for the exact ω
    for ki in 1:Nk
        ω = ωs_f[ki]
        ap = zeros(ComplexF64, 6)
        am = zeros(ComplexF64, 6)

        for m in 1:N_t
            t = t_grid[m]
            @views T_vec = T_vals[:, ki, m] # Avoids allocating a new vector

            ap .+= T_vec .* cis(+ω * t)
            am .+= T_vec .* cis(-ω * t)
        end

        ap .*= Δt
        am .*= Δt

        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

# ── ∂iϕ∂jϕ_AzimuthalReduction · ConstantWeight ──────────────────────────────

"""
    QuadGkConstantTimeIntegratedTwoPointStressEnergyTensor(ωs, snapshot, space,
        boundary_condition, strategy; ΔV=1.0, bubble_indices=:,
        rtol=1e-3, atol=0.0, maxevals=typemax(Int))

Computes the time-integrated two-point correlator without temporal decoherence:

  δ^const_{ij,lm} = A⁰_{ij}(k) conj(A⁰_{lm}(k)) / V

where  A⁰_{ij}(k) = ∫ T̃_{ij}(t, k) dt  (no oscillating phase factor).

Returns a 6×6×Nk array of ComplexF64.
"""
function QuadGkConstantTimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::∂iϕ∂jϕ_AzimuthalReduction;
    ΔV::Float64 = 1.0,
    bubble_indices = :,
    rtol::Real = 1e-3,
    atol::Real = 0.0,
    maxevals::Int = typemax(Int)
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    V_inv   = 1.0 / volume(space)

    function integrand(t::Float64)
        bubbles = current_bubbles(snapshot, t)
        T = ∂iϕ∂jϕ(ωs_f, bubbles, space, boundary_condition,
                     strategy.plan, strategy.buffer; ΔV=ΔV, bubble_indices=bubble_indices)
        return vec(T)
    end

    a0_vec, _ = quadgk(integrand, t_start, t_end;
                        rtol=rtol, atol=atol, maxevals=maxevals)

    A0     = reshape(a0_vec, 6, Nk)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        a = A0[:, ki]
        result[:, :, ki] = (a * a') .* V_inv
    end

    return result
end

function UniformGridConstantIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::∂iϕ∂jϕ_AzimuthalReduction;
    N_t::Int,
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    Δt      = (t_end - t_start) / (N_t - 1)
    t_grid  = t_start .+ (0:N_t-1) .* Δt

    T_vals = Array{ComplexF64, 3}(undef, 6, Nk, N_t)
    for m in 1:N_t
        bubbles = current_bubbles(snapshot, Float64(t_grid[m]))
        T_vals[:, :, m] = ∂iϕ∂jϕ(ωs_f, bubbles, space, boundary_condition,
                                    strategy.plan, strategy.buffer; ΔV=ΔV,
                                    bubble_indices=bubble_indices)
    end

    # Trapezoidal weights: endpoints get half weight
    T_vals[:, :, 1]   .*= 0.5
    T_vals[:, :, end] .*= 0.5

    V_inv  = 1.0 / volume(space)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)

    for ki in 1:Nk
        a0 = zeros(ComplexF64, 6)
        for m in 1:N_t
            @views T_vec = T_vals[:, ki, m]
            a0 .+= T_vec
        end
        a0 .*= Δt
        result[:, :, ki] = (a0 * a0') .* V_inv
    end

    return result
end

# ── Unified API · ∂iϕ∂jϕ_AzimuthalReduction ─────────────────────────────────

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::∂iϕ∂jϕ_AzimuthalReduction,
    time_strategy::TimeIntegrationStrategy,
    temporal_weight::TemporalWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    throw(ArgumentError("Unsupported time_strategy: $(typeof(time_strategy))"))
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::∂iϕ∂jϕ_AzimuthalReduction,
    time_strategy::QuadGKIntegration,
    ::CosineWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    QuadGkCosineTimeIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        ΔV=ΔV, bubble_indices=bubble_indices, rtol=time_strategy.rtol,
        atol=time_strategy.atol, maxevals=time_strategy.maxevals
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::∂iϕ∂jϕ_AzimuthalReduction,
    time_strategy::UniformGridIntegration,
    ::CosineWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    UniformGridCosineIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        N_t=time_strategy.N_t, ΔV=ΔV, bubble_indices=bubble_indices
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::∂iϕ∂jϕ_AzimuthalReduction,
    time_strategy::QuadGKIntegration,
    ::ConstantWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    QuadGkConstantTimeIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        ΔV=ΔV, bubble_indices=bubble_indices, rtol=time_strategy.rtol,
        atol=time_strategy.atol, maxevals=time_strategy.maxevals
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::∂iϕ∂jϕ_AzimuthalReduction,
    time_strategy::UniformGridIntegration,
    ::ConstantWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    UniformGridConstantIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        N_t=time_strategy.N_t, ΔV=ΔV, bubble_indices=bubble_indices
    )
end

# ── T_ij (kinetic + potential) strategy ──────────────────────────────────────

function TwoPointStressEnergyTensor(t1::Real, t2::Real, ωs::AbstractVector{<:Real},
                                     snapshot::BubblesSnapShot, space::AbstractSpace,
                                     boundary_condition::BoundaryCondition,
                                     strategy::T_ij_AzimuthalReduction;
                                     ΔV::Float64=1.0,
                                     bubble_indices=:)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    bubbles1 = current_bubbles(snapshot, Float64(t1))
    bubbles2 = current_bubbles(snapshot, Float64(t2))
    T1 = T_ij(ωs_f, bubbles1, space, boundary_condition, strategy.plan, strategy.kernel;
               ΔV=ΔV, bubble_indices=bubble_indices)
    T2 = T_ij(ωs_f, bubbles2, space, boundary_condition, strategy.plan, strategy.kernel;
               ΔV=ΔV, bubble_indices=bubble_indices)
    V_inv = 1.0 / volume(space)
    return reshape(T1, 6, 1, :) .* conj.(reshape(T2, 1, 6, :)) .* V_inv
end

# ── T_ij_AzimuthalReduction · CosineWeight ───────────────────────────────────

function QuadGkCosineTimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::T_ij_AzimuthalReduction;
    ΔV::Float64 = 1.0,
    bubble_indices = :,
    rtol::Real = 1e-3,
    atol::Real = 0.0,
    maxevals::Int = typemax(Int)
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    V_inv   = 1.0 / volume(space)

    function integrand(t::Float64)
        bubbles = current_bubbles(snapshot, t)
        T = T_ij(ωs_f, bubbles, space, boundary_condition,
                  strategy.plan, strategy.kernel; ΔV=ΔV, bubble_indices=bubble_indices)
        phases = cis.(ωs_f .* t)
        return vcat(vec(T .* reshape(phases,        1, Nk)),
                    vec(T .* reshape(conj.(phases), 1, Nk)))
    end

    combined, _ = quadgk(integrand, t_start, t_end;
                         rtol=rtol, atol=atol, maxevals=maxevals)

    A_plus  = reshape(combined[1:6Nk],     6, Nk)
    A_minus = reshape(combined[6Nk+1:end], 6, Nk)

    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        ap = A_plus[:,  ki]
        am = A_minus[:, ki]
        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

function UniformGridCosineIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::T_ij_AzimuthalReduction;
    N_t::Int,
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    Δt      = (t_end - t_start) / (N_t - 1)
    t_grid  = t_start .+ (0:N_t-1) .* Δt

    T_vals = Array{ComplexF64, 3}(undef, 6, Nk, N_t)
    for m in 1:N_t
        bubbles = current_bubbles(snapshot, Float64(t_grid[m]))
        T_vals[:, :, m] = T_ij(ωs_f, bubbles, space, boundary_condition,
                                strategy.plan, strategy.kernel; ΔV=ΔV,
                                bubble_indices=bubble_indices)
    end

    T_vals[:, :, 1]   .*= 0.5
    T_vals[:, :, end] .*= 0.5

    V_inv  = 1.0 / volume(space)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)

    for ki in 1:Nk
        ω = ωs_f[ki]
        ap = zeros(ComplexF64, 6)
        am = zeros(ComplexF64, 6)

        for m in 1:N_t
            t = t_grid[m]
            @views T_vec = T_vals[:, ki, m]
            ap .+= T_vec .* cis(+ω * t)
            am .+= T_vec .* cis(-ω * t)
        end

        ap .*= Δt
        am .*= Δt

        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

# ── T_ij_AzimuthalReduction · ConstantWeight ─────────────────────────────────

function QuadGkConstantTimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::T_ij_AzimuthalReduction;
    ΔV::Float64 = 1.0,
    bubble_indices = :,
    rtol::Real = 1e-3,
    atol::Real = 0.0,
    maxevals::Int = typemax(Int)
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    V_inv   = 1.0 / volume(space)

    function integrand(t::Float64)
        bubbles = current_bubbles(snapshot, t)
        T = T_ij(ωs_f, bubbles, space, boundary_condition,
                  strategy.plan, strategy.kernel; ΔV=ΔV, bubble_indices=bubble_indices)
        return vec(T)
    end

    a0_vec, _ = quadgk(integrand, t_start, t_end;
                        rtol=rtol, atol=atol, maxevals=maxevals)

    A0     = reshape(a0_vec, 6, Nk)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        a = A0[:, ki]
        result[:, :, ki] = (a * a') .* V_inv
    end

    return result
end

function UniformGridConstantIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    strategy::T_ij_AzimuthalReduction;
    N_t::Int,
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    ωs_f = collect(Float64, ωs)
    Nk   = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    t_start = minimum(nuc[:time] for nuc in snapshot.nucleations)
    t_end   = snapshot.t
    Δt      = (t_end - t_start) / (N_t - 1)
    t_grid  = t_start .+ (0:N_t-1) .* Δt

    T_vals = Array{ComplexF64, 3}(undef, 6, Nk, N_t)
    for m in 1:N_t
        bubbles = current_bubbles(snapshot, Float64(t_grid[m]))
        T_vals[:, :, m] = T_ij(ωs_f, bubbles, space, boundary_condition,
                                strategy.plan, strategy.kernel; ΔV=ΔV,
                                bubble_indices=bubble_indices)
    end

    T_vals[:, :, 1]   .*= 0.5
    T_vals[:, :, end] .*= 0.5

    V_inv  = 1.0 / volume(space)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)

    for ki in 1:Nk
        a0 = zeros(ComplexF64, 6)
        for m in 1:N_t
            @views T_vec = T_vals[:, ki, m]
            a0 .+= T_vec
        end
        a0 .*= Δt
        result[:, :, ki] = (a0 * a0') .* V_inv
    end

    return result
end

# ── Unified API · T_ij_AzimuthalReduction ────────────────────────────────────

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::T_ij_AzimuthalReduction,
    time_strategy::TimeIntegrationStrategy,
    temporal_weight::TemporalWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    throw(ArgumentError("Unsupported time_strategy: $(typeof(time_strategy))"))
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::T_ij_AzimuthalReduction,
    time_strategy::QuadGKIntegration,
    ::CosineWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    QuadGkCosineTimeIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        ΔV=ΔV, bubble_indices=bubble_indices, rtol=time_strategy.rtol,
        atol=time_strategy.atol, maxevals=time_strategy.maxevals
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::T_ij_AzimuthalReduction,
    time_strategy::UniformGridIntegration,
    ::CosineWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    UniformGridCosineIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        N_t=time_strategy.N_t, ΔV=ΔV, bubble_indices=bubble_indices
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::T_ij_AzimuthalReduction,
    time_strategy::QuadGKIntegration,
    ::ConstantWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    QuadGkConstantTimeIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        ΔV=ΔV, bubble_indices=bubble_indices, rtol=time_strategy.rtol,
        atol=time_strategy.atol, maxevals=time_strategy.maxevals
    )
end

function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::AbstractSpace,
    boundary_condition::BoundaryCondition,
    spatial_strategy::T_ij_AzimuthalReduction,
    time_strategy::UniformGridIntegration,
    ::ConstantWeight;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)::Array{ComplexF64, 3}
    UniformGridConstantIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, spatial_strategy;
        N_t=time_strategy.N_t, ΔV=ΔV, bubble_indices=bubble_indices
    )
end

# ── RayCastingSphericalQuadrature ────────────────────────────────────────────

"""
    TimeIntegratedTwoPointStressEnergyTensor(ωs, snapshot, space, boundary_condition,
        strategy::RayCastingSphericalQuadrature, weight::CosineWeight;
        term=KineticTerm(), ΔV=1.0, bubble_indices=:)

Ray-casting two-point correlator with cosine temporal weight:

  (1/2V) [ A⁺_{ij}(k) conj(A⁺_{lm}(k)) + A⁻_{ij}(k) conj(A⁻_{lm}(k)) ]

where A± are computed via ray casting with analytic τ-integrals. Pass `term=TotalStressTerm()`
to include both kinetic and potential contributions.

Returns a 6×6×Nk array of ComplexF64.
"""
function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    boundary_condition::Periodic,
    strategy::RayCastingSphericalQuadrature,
    ::CosineWeight;
    term::StressTensorTerm = KineticTerm(),
    ΔV::Float64 = 1.0,
    bubble_indices = :
)
    ωs_f = ωs isa AbstractRange ? ωs : collect(Float64, ωs)
    Nk = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    V_inv = 1.0 / volume(space)
    A_plus, A_minus = amplitudes(
        ray_T_ij(ωs_f, snapshot, space, boundary_condition,
                 term, CosineWeight(), strategy.quadrature;
                 ΔV=ΔV, bubble_indices=bubble_indices)
    )

    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        ap = A_plus[:,  ki]
        am = A_minus[:, ki]
        result[:, :, ki] = (ap * ap' .+ am * am') .* (V_inv / 2)
    end

    return result
end

"""
    TimeIntegratedTwoPointStressEnergyTensor(ωs, snapshot, space, boundary_condition,
        strategy::RayCastingSphericalQuadrature, weight::ConstantWeight;
        term=KineticTerm(), ΔV=1.0, bubble_indices=:)

Ray-casting two-point correlator with constant temporal weight:

  (1/V) A⁰_{ij}(k) conj(A⁰_{lm}(k))

where A⁰ is computed via ray casting with analytic τ-integrals and no oscillating phase.
Pass `term=TotalStressTerm()` to include both kinetic and potential contributions.

Returns a 6×6×Nk array of ComplexF64.
"""
function TimeIntegratedTwoPointStressEnergyTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    boundary_condition::Periodic,
    strategy::RayCastingSphericalQuadrature,
    ::ConstantWeight;
    term::StressTensorTerm = KineticTerm(),
    ΔV::Float64 = 1.0,
    bubble_indices = :
)
    ωs_f = ωs isa AbstractRange ? ωs : collect(Float64, ωs)
    Nk = length(ωs_f)

    isempty(snapshot.nucleations) && return zeros(ComplexF64, 6, 6, Nk)

    V_inv = 1.0 / volume(space)
    A = amplitudes(
        ray_T_ij(ωs_f, snapshot, space, boundary_condition,
                 term, ConstantWeight(), strategy.quadrature;
                 ΔV=ΔV, bubble_indices=bubble_indices)
    )

    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        a = A[:, ki]
        result[:, :, ki] = (a * a') .* V_inv
    end

    return result
end

"""
    RayCastingCosineTimeIntegratedTwoPointStressTensor(ωs, snapshot, space,
        boundary_condition, strategy; ΔV=1.0, bubble_indices=:)

Convenience wrapper for `TimeIntegratedTwoPointStressEnergyTensor` with `CosineWeight`
and `KineticTerm`. See that function for full documentation.
"""
function RayCastingCosineTimeIntegratedTwoPointStressTensor(
    ωs::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    boundary_condition::Periodic,
    strategy::RayCastingSphericalQuadrature;
    ΔV::Float64 = 1.0,
    bubble_indices = :
)
    return TimeIntegratedTwoPointStressEnergyTensor(
        ωs, snapshot, space, boundary_condition, strategy, CosineWeight();
        term=KineticTerm(), ΔV=ΔV, bubble_indices=bubble_indices
    )
end

# ── Cross-product helpers ─────────────────────────────────────────────────────

function _ray_cross_product(
    acc_α::CosineAccumulant, acc_β::CosineAccumulant,
    V_inv::Float64, Nk::Int,
)
    A_plus_α,  A_minus_α  = amplitudes(acc_α)
    A_plus_β,  A_minus_β  = amplitudes(acc_β)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        ap_α = A_plus_α[:,  ki];  am_α = A_minus_α[:, ki]
        ap_β = A_plus_β[:,  ki];  am_β = A_minus_β[:, ki]
        result[:, :, ki] = (ap_α * ap_β' .+ am_α * am_β') .* (V_inv / 2)
    end
    return result
end

function _ray_cross_product(
    acc_α::ConstantAccumulant, acc_β::ConstantAccumulant,
    V_inv::Float64, Nk::Int,
)
    A_α = amplitudes(acc_α)
    A_β = amplitudes(acc_β)
    result = Array{ComplexF64, 3}(undef, 6, 6, Nk)
    for ki in 1:Nk
        result[:, :, ki] = (A_α[:, ki] * A_β[:, ki]') .* V_inv
    end
    return result
end

# ── Pieces API · RayCastingSphericalQuadrature ────────────────────────────────

"""
    TimeIntegratedTwoPointStressEnergyTensorPieces(ks, snapshot, space,
        boundary_condition, strategy, weight;
        terms=(:K=>KineticTerm(), :V=>PotentialTerm()),
        ΔV=1.0, bubble_indices=:, v=1.0)

Compute the time-integrated two-point correlator split by source term.

Runs the ray-casting geometry loop once and returns a nested `NamedTuple` of
all cross-products between terms. For `terms = (:K => KineticTerm(), :V =>
PotentialTerm())` the result has the shape

    (K = (K = arr_KK, V = arr_KV),
     V = (K = arr_VK, V = arr_VV))

where `arr_αβ` is the 6×6×Nk array for the α×β cross-product:

  CosineWeight:   (1/2V) [A⁺_α conj(A⁺_β)' + A⁻_α conj(A⁻_β)']
  ConstantWeight: (1/V)   A⁰_α conj(A⁰_β)'

The total correlator equals the sum over all (α,β) pairs.
"""
function TimeIntegratedTwoPointStressEnergyTensorPieces(
    ks::AbstractVector{<:Real},
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    boundary_condition::Periodic,
    strategy::RayCastingSphericalQuadrature,
    weight::W;
    terms = (:K => KineticTerm(), :V => PotentialTerm()),
    ΔV::Float64 = 1.0,
    bubble_indices = :,
    v::Float64 = 1.0,
) where {W<:TemporalWeight}
    ks_f  = ks isa AbstractRange ? ks : collect(Float64, ks)
    Nk    = length(ks_f)
    V_inv = 1.0 / volume(space)

    accs  = ray_T_ij(ks_f, snapshot, space, boundary_condition,
                     terms, weight, strategy.quadrature;
                     ΔV=ΔV, bubble_indices=bubble_indices, v=v)

    names = map(first, terms)
    return NamedTuple{names}(map(names) do name_α
        acc_α = accs[name_α]
        NamedTuple{names}(map(name_β -> _ray_cross_product(acc_α, accs[name_β], V_inv, Nk), names))
    end)
end

end
