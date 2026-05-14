import Base.resize!

include("SinCosGrid.jl")

# ============================================================
# Mode workspace
# ============================================================

abstract type ModeWorkspace{W<:TemporalWeight} end

mutable struct CosineModeWorkspace <: ModeWorkspace{CosineWeight}
    phase_plus::Vector{ComplexF64}
    phase_minus::Vector{ComplexF64}

    S_plus::Vector{Float64}
    C_plus::Vector{Float64}
    S_minus::Vector{Float64}
    C_minus::Vector{Float64}
end

mutable struct ConstantModeWorkspace <: ModeWorkspace{ConstantWeight}
    phase::Vector{ComplexF64}

    S::Vector{Float64}
    C::Vector{Float64}
end

function ModeWorkspace(::CosineWeight, Nk::Int)
    return CosineModeWorkspace(
        Vector{ComplexF64}(undef, Nk),
        Vector{ComplexF64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
    )
end

function ModeWorkspace(::ConstantWeight, Nk::Int)
    return ConstantModeWorkspace(
        Vector{ComplexF64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
    )
end

function resize!(ws::CosineModeWorkspace, Nk::Int)
    resize!(ws.phase_plus, Nk)
    resize!(ws.phase_minus, Nk)
    resize!(ws.S_plus, Nk)
    resize!(ws.C_plus, Nk)
    resize!(ws.S_minus, Nk)
    resize!(ws.C_minus, Nk)
    return ws
end

function resize!(ws::ConstantModeWorkspace, Nk::Int)
    resize!(ws.phase, Nk)
    resize!(ws.S, Nk)
    resize!(ws.C, Nk)
    return ws
end

function prepare_source_modes!(
    ws::CosineModeWorkspace,
    ::CosineWeight,
    ks::AbstractVector{<:Real},
    source_nucleation::Nucleation,
)
    resize!(ws, length(ks))

    t_i = source_nucleation[:time]
    z_i = source_nucleation[:site].coordinates[3]

    @inbounds for q in eachindex(ks)
        ws.phase_plus[q] = cis(ks[q] * (t_i - z_i))
        ws.phase_minus[q] = cis(-ks[q] * (t_i + z_i))
    end

    return nothing
end

function prepare_source_modes!(
    ws::ConstantModeWorkspace,
    ::ConstantWeight,
    ks::AbstractVector{<:Real},
    source_nucleation::Nucleation,
)
    resize!(ws, length(ks))

    z_i = source_nucleation[:site].coordinates[3]

    @inbounds for q in eachindex(ks)
        ws.phase[q] = cis(-ks[q] * z_i)
    end

    return nothing
end

# ============================================================
# Tensor component helpers
# ============================================================

@inline function outer_prod(n1::Float64, n2::Float64, n3::Float64)
    return SVector{6,Float64}(
        n1*n1,
        n1*n2,
        n1*n3,
        n2*n2,
        n2*n3,
        n3*n3,
    )
end

@inline function outer_prod(n̂::SVector{3,Float64})
    n1, n2, n3 = n̂
    return outer_prod(n1, n2, n3)
end

@inline function accumulate_tensor_components!(
    A::Matrix{ComplexF64},
    q::Int,
    amp::ComplexF64,
    nᵢnⱼ::SVector{6,Float64},
)
    @inbounds for I in 1:6
        A[I, q] = muladd(amp, nᵢnⱼ[I], A[I, q])
    end
    return nothing
end

@inline function accumulate_diagonal_components!(
    A::Matrix{ComplexF64},
    q::Int,
    amp::ComplexF64,
)
    @inbounds begin
        A[1, q] += amp
        A[4, q] += amp
        A[6, q] += amp
    end
    return nothing
end

# ============================================================
# CosineWeight accumulation
# ============================================================

@inline function add_kinetic_mode!(
    acc::CosineAccumulant,
    ws::CosineModeWorkspace,
    ::CosineWeight,
    ks,
    q::Int,
    τ_stop::Float64,
    nᵢnⱼ::SVector{6,Float64},
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
)
    k = Float64(ks[q])
    prefactor = marker_weight * (ΔV / 3.0) * v^3

    # A±_ij += w (ΔV/3) v^3 n_i n_j e^{-ikz_i} e^{±ikt_i}
    #          ∫_0^τ τ'^3 exp(ik(±1 - v n_z)τ') dτ'
    amp_plus = prefactor * ws.phase_plus[q] *
               I3_zero_lower(k * (1.0 - v*n3), τ_stop)
    amp_minus = prefactor * ws.phase_minus[q] *
                I3_zero_lower(k * (-1.0 - v*n3), τ_stop)

    accumulate_tensor_components!(acc.A_plus, q, amp_plus, nᵢnⱼ)
    accumulate_tensor_components!(acc.A_minus, q, amp_minus, nᵢnⱼ)
    return nothing
end

@inline function add_potential_mode!(
    acc::CosineAccumulant,
    ws::CosineModeWorkspace,
    ::CosineWeight,
    ks,
    q::Int,
    τ_stop::Float64,
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
)
    k = Float64(ks[q])

    iszero(k) && throw(ArgumentError("PotentialTerm with k = 0 is not supported by ray_T_ij"))

    prefactor = marker_weight * im * ΔV * v^2 * n3 / k

    # A±_diag += w i ΔV v^2 n_z/k e^{-ikz_i} e^{±ikt_i}
    #            ∫_0^τ τ'^2 exp(ik(±1 - v n_z)τ') dτ'
    amp_plus = prefactor * ws.phase_plus[q] *
               I2_zero_lower(k * (1.0 - v*n3), τ_stop)
    amp_minus = prefactor * ws.phase_minus[q] *
                I2_zero_lower(k * (-1.0 - v*n3), τ_stop)

    accumulate_diagonal_components!(acc.A_plus, q, amp_plus)
    accumulate_diagonal_components!(acc.A_minus, q, amp_minus)
    return nothing
end

# ============================================================
# ConstantWeight accumulation
# ============================================================

@inline function add_kinetic_mode!(
    acc::ConstantAccumulant,
    ws::ConstantModeWorkspace,
    ::ConstantWeight,
    ks,
    q::Int,
    τ_stop::Float64,
    nᵢnⱼ::SVector{6,Float64},
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
)
    k = Float64(ks[q])

    # A_ij += w (ΔV/3) v^3 n_i n_j e^{-ikz_i}
    #         ∫_0^τ τ'^3 exp(-ik v n_z τ') dτ'
    amp = marker_weight * (ΔV / 3.0) * v^3 * ws.phase[q] *
          I3_zero_lower(-k * v*n3, τ_stop)

    accumulate_tensor_components!(acc.A, q, amp, nᵢnⱼ)
    return nothing
end

function add_potential_mode!(
    acc::ConstantAccumulant,
    ws::ConstantModeWorkspace,
    ::ConstantWeight,
    ks,
    q::Int,
    τ_stop::Float64,
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
)
    k = Float64(ks[q])
    iszero(k) && throw(ArgumentError("PotentialTerm with k = 0 is not supported by ray_T_ij"))

    prefactor = marker_weight * im * ΔV * v^2 * n3 / k

    # A_diag += w i ΔV v^2 n_z/k e^{-ikz_i}
    #           ∫_0^τ τ'^2 exp(-ik v n_z τ') dτ'
    amp = prefactor * ws.phase[q] * I2_zero_lower(-k * v*n3, τ_stop)

    accumulate_diagonal_components!(acc.A, q, amp)
    return nothing
end

# ============================================================
# Term dispatch
# ============================================================

@inline function add_term_mode!(
    acc::Accumulant{W},
    ws::ModeWorkspace{W},
    weight::W,
    ::KineticTerm,
    ks,
    q::Int,
    τ_stop::Float64,
    nᵢnⱼ::SVector{6,Float64},
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
) where {W<:TemporalWeight}
    return add_kinetic_mode!(acc, ws, weight, ks, q, τ_stop, nᵢnⱼ, n3, marker_weight, ΔV, v)
end

@inline function add_term_mode!(
    acc::Accumulant{W},
    ws::ModeWorkspace{W},
    weight::W,
    ::PotentialTerm,
    ks,
    q::Int,
    τ_stop::Float64,
    nᵢnⱼ::SVector{6,Float64},
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
) where {W<:TemporalWeight}
    return add_potential_mode!(acc, ws, weight, ks, q, τ_stop, n3, marker_weight, ΔV, v)
end

@inline function add_term_mode!(
    acc::Accumulant{W},
    ws::ModeWorkspace{W},
    weight::W,
    ::TotalStressTerm,
    ks,
    q::Int,
    τ_stop::Float64,
    nᵢnⱼ::SVector{6,Float64},
    n3::Float64,
    marker_weight::Float64,
    ΔV::Float64,
    v::Float64,
) where {W<:TemporalWeight}
    add_kinetic_mode!(acc, ws, weight, ks, q, τ_stop, nᵢnⱼ, n3, marker_weight, ΔV, v)
    add_potential_mode!(acc, ws, weight, ks, q, τ_stop, n3, marker_weight, ΔV, v)
    return nothing
end

function accumulate_marker_modes!(
    acc::Accumulant{W},
    weight::W,
    term::StressTensorTerm,
    ks::AbstractVector{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::ModeWorkspace{W},
    ΔV::Float64,
    v::Float64,
) where {W<:TemporalWeight}
    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]

    @inbounds for q in eachindex(ks)
        add_term_mode!(acc, ws, weight, term, ks, q, τ_stop, nᵢnⱼ, n3, marker_weight, ΔV, v)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::CosineAccumulant,
    ::CosineWeight,
    ::PotentialTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::CosineModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    n3 = n̂[3]

    c_plus  = 1.0 - v*n3
    c_minus = -1.0 - v*n3

    compute_sincos_grid!(ws.S_plus,  ws.C_plus,  ks, c_plus,  τ_stop)
    compute_sincos_grid!(ws.S_minus, ws.C_minus, ks, c_minus, τ_stop)

    prefactor = marker_weight * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        pf = prefactor / k

        amp_plus  = pf * ws.phase_plus[q]  * I2_zero_lower_from_sincos(k*c_plus,  τ_stop, ws.S_plus[q],  ws.C_plus[q])
        amp_minus = pf * ws.phase_minus[q] * I2_zero_lower_from_sincos(k*c_minus, τ_stop, ws.S_minus[q], ws.C_minus[q])

        accumulate_diagonal_components!(acc.A_plus,  q, amp_plus)
        accumulate_diagonal_components!(acc.A_minus, q, amp_minus)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::PotentialTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::ConstantModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    n3 = n̂[3]
    c_val = -v * n3

    compute_sincos_grid!(ws.S, ws.C, ks, c_val, τ_stop)

    prefactor = marker_weight * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        amp = (prefactor / k) * ws.phase[q] * I2_zero_lower_from_sincos(k*c_val, τ_stop, ws.S[q], ws.C[q])

        accumulate_diagonal_components!(acc.A, q, amp)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::CosineAccumulant,
    ::CosineWeight,
    ::TotalStressTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::CosineModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]

    c_plus  = 1.0 - v*n3
    c_minus = -1.0 - v*n3

    compute_sincos_grid!(ws.S_plus,  ws.C_plus,  ks, c_plus,  τ_stop)
    compute_sincos_grid!(ws.S_minus, ws.C_minus, ks, c_minus, τ_stop)

    kinetic_pf  = marker_weight * (ΔV / 3.0) * v^3
    potential_pf = marker_weight * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        S_p, C_p = ws.S_plus[q],  ws.C_plus[q]
        S_m, C_m = ws.S_minus[q], ws.C_minus[q]

        amp_plus_k  = kinetic_pf * ws.phase_plus[q]  * I3_zero_lower_from_sincos(k*c_plus,  τ_stop, S_p, C_p)
        amp_minus_k = kinetic_pf * ws.phase_minus[q] * I3_zero_lower_from_sincos(k*c_minus, τ_stop, S_m, C_m)
        accumulate_tensor_components!(acc.A_plus,  q, amp_plus_k,  nᵢnⱼ)
        accumulate_tensor_components!(acc.A_minus, q, amp_minus_k, nᵢnⱼ)

        pf_v = potential_pf / k
        amp_plus_v  = pf_v * ws.phase_plus[q]  * I2_zero_lower_from_sincos(k*c_plus,  τ_stop, S_p, C_p)
        amp_minus_v = pf_v * ws.phase_minus[q] * I2_zero_lower_from_sincos(k*c_minus, τ_stop, S_m, C_m)
        accumulate_diagonal_components!(acc.A_plus,  q, amp_plus_v)
        accumulate_diagonal_components!(acc.A_minus, q, amp_minus_v)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::TotalStressTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::ConstantModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]
    c_val = -v * n3

    compute_sincos_grid!(ws.S, ws.C, ks, c_val, τ_stop)

    kinetic_pf  = marker_weight * (ΔV / 3.0) * v^3
    potential_pf = marker_weight * im * ΔV * v^2 * n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        S, C = ws.S[q], ws.C[q]

        amp_k = kinetic_pf * ws.phase[q] * I3_zero_lower_from_sincos(k*c_val, τ_stop, S, C)
        accumulate_tensor_components!(acc.A, q, amp_k, nᵢnⱼ)

        amp_v = (potential_pf / k) * ws.phase[q] * I2_zero_lower_from_sincos(k*c_val, τ_stop, S, C)
        accumulate_diagonal_components!(acc.A, q, amp_v)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::ConstantAccumulant,
    ::ConstantWeight,
    ::KineticTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::ConstantModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]

    c_val = -v * n3

    compute_sincos_grid!(ws.S, ws.C, ks, c_val, τ_stop)

    prefactor = marker_weight * (ΔV / 3.0) * v^3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        amp = prefactor * ws.phase[q] *
              I3_zero_lower_from_sincos(k * c_val, τ_stop, ws.S[q], ws.C[q])

        accumulate_tensor_components!(acc.A, q, amp, nᵢnⱼ)
    end

    return nothing
end

function accumulate_marker_modes!(
    acc::CosineAccumulant,
    ::CosineWeight,
    ::KineticTerm,
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂::SVector{3,Float64},
    marker_weight::Float64,
    ws::CosineModeWorkspace,
    ΔV::Float64,
    v::Float64,
)
    nᵢnⱼ = outer_prod(n̂)
    n3 = n̂[3]

    c_plus = 1.0 - v*n3
    c_minus = -1.0 - v*n3

    compute_sincos_grid!(ws.S_plus, ws.C_plus, ks, c_plus, τ_stop)
    compute_sincos_grid!(ws.S_minus, ws.C_minus, ks, c_minus, τ_stop)

    prefactor = marker_weight * (ΔV / 3.0) * v^3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        amp_plus = prefactor * ws.phase_plus[q] *
                   I3_zero_lower_from_sincos(k*c_plus, τ_stop, ws.S_plus[q], ws.C_plus[q])
        amp_minus = prefactor * ws.phase_minus[q] *
                    I3_zero_lower_from_sincos(k*c_minus, τ_stop, ws.S_minus[q], ws.C_minus[q])

        accumulate_tensor_components!(acc.A_plus, q, amp_plus, nᵢnⱼ)
        accumulate_tensor_components!(acc.A_minus, q, amp_minus, nᵢnⱼ)
    end

    return nothing
end
