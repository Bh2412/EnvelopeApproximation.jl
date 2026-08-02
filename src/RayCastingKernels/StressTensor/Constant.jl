mutable struct ConstantModeWorkspace <: ModeWorkspace{ConstantWeight}
    phase::Vector{ComplexF64}

    S::Vector{Float64}
    C::Vector{Float64}
end

function ModeWorkspace(::ConstantWeight, Nk::Int)
    return ConstantModeWorkspace(
        Vector{ComplexF64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
    )
end

function resize!(ws::ConstantModeWorkspace, Nk::Int)
    resize!(ws.phase, Nk)
    resize!(ws.S, Nk)
    resize!(ws.C, Nk)
    return ws
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

function prepare_source_modes!(
    ws::ConstantModeWorkspace,
    ::ConstantWeight,
    ks::AbstractVector{<:Real},
    source::EnvelopeSource,
)
    resize!(ws, length(ks))

    z_i = source.center[3]

    @inbounds for q in eachindex(ks)
        ws.phase[q] = cis(-ks[q] * z_i)
    end

    return nothing
end

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
