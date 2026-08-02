mutable struct ComplexExponentialModeWorkspace{N} <: ModeWorkspace{ComplexExponential{N}}
    phase::Matrix{ComplexF64}
end

function ModeWorkspace(::ComplexExponential{N}, Nk::Int) where {N}
    return ComplexExponentialModeWorkspace{N}(
        Matrix{ComplexF64}(undef, Nk, N),
    )
end

function resize!(ws::ComplexExponentialModeWorkspace{N}, Nk::Int) where {N}
    size(ws.phase) == (Nk, N) || (ws.phase = Matrix{ComplexF64}(undef, Nk, N))
    return ws
end

function prepare_source_modes!(
    ws::ComplexExponentialModeWorkspace,
    weight::ComplexExponential,
    ks::AbstractVector{<:Real},
    source_nucleation::Nucleation,
)
    resize!(ws, length(ks))

    t_i = source_nucleation[:time]
    z_i = source_nucleation[:site].coordinates[3]

    @inbounds for p in eachindex(weight.ωs), q in eachindex(ks)
        ws.phase[q, p] = cis(weight.ωs[p] * t_i - ks[q] * z_i)
    end

    return nothing
end

function prepare_source_modes!(
    ws::ComplexExponentialModeWorkspace,
    weight::ComplexExponential,
    ks::AbstractVector{<:Real},
    source::EnvelopeSource,
)
    resize!(ws, length(ks))

    t_i = source.time
    z_i = source.center[3]

    @inbounds for p in eachindex(weight.ωs), q in eachindex(ks)
        ws.phase[q, p] = cis(weight.ωs[p] * t_i - ks[q] * z_i)
    end

    return nothing
end

@inline function accumulate_tensor_components!(
    A::Array{ComplexF64,3},
    q::Int,
    p::Int,
    amp::ComplexF64,
    nᵢnⱼ::SVector{6,Float64},
)
    @inbounds for I in 1:6
        A[I, q, p] = muladd(amp, nᵢnⱼ[I], A[I, q, p])
    end
    return nothing
end

@inline function accumulate_diagonal_components!(
    A::Array{ComplexF64,3},
    q::Int,
    p::Int,
    amp::ComplexF64,
)
    @inbounds begin
        A[1, q, p] += amp
        A[4, q, p] += amp
        A[6, q, p] += amp
    end
    return nothing
end

@inline function add_kinetic_mode!(
    acc::ComplexExponentialAccumulant,
    ws::ComplexExponentialModeWorkspace,
    weight::ComplexExponential,
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

    @inbounds for p in eachindex(weight.ωs)
        ω = weight.ωs[p]
        amp = prefactor * ws.phase[q, p] *
              I3_zero_lower(ω - k*v*n3, τ_stop)
        accumulate_tensor_components!(acc.A, q, p, amp, nᵢnⱼ)
    end

    return nothing
end

@inline function add_potential_mode!(
    acc::ComplexExponentialAccumulant,
    ws::ComplexExponentialModeWorkspace,
    weight::ComplexExponential,
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

    @inbounds for p in eachindex(weight.ωs)
        ω = weight.ωs[p]
        amp = prefactor * ws.phase[q, p] *
              I2_zero_lower(ω - k*v*n3, τ_stop)
        accumulate_diagonal_components!(acc.A, q, p, amp)
    end

    return nothing
end
