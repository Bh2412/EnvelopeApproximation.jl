import Base.resize!

include("SinCosGrid.jl")

# ============================================================
# Workspace
# ============================================================

mutable struct SourceModeWorkspace
    phase_plus::Vector{ComplexF64}
    phase_minus::Vector{ComplexF64}

    S_plus::Vector{Float64}
    C_plus::Vector{Float64}
    S_minus::Vector{Float64}
    C_minus::Vector{Float64}
end

SourceModeWorkspace(Nk::Int) =
    SourceModeWorkspace(
        Vector{ComplexF64}(undef, Nk),
        Vector{ComplexF64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
        Vector{Float64}(undef, Nk),
    )

function resize!(ws::SourceModeWorkspace, Nk::Int)
    resize!(ws.phase_plus, Nk)
    resize!(ws.phase_minus, Nk)

    resize!(ws.S_plus, Nk)
    resize!(ws.C_plus, Nk)
    resize!(ws.S_minus, Nk)
    resize!(ws.C_minus, Nk)

    return ws
end

# ============================================================
# Source-dependent k phases
# ============================================================


function prepare_source_phases!(
    ws::SourceModeWorkspace,
    ks::AbstractVector{<:Real},
    t_i::Float64,
    zi::Float64,
    amp_base::Float64,
)
    resize!(ws, length(ks))

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        ws.phase_plus[q]  = amp_base * cis( k * ( t_i - zi))
        ws.phase_minus[q] = amp_base * cis(-k * ( t_i + zi))
    end

    return nothing
end

# ============================================================
# Tensor component accumulation
# ============================================================

@inline function accumulate_tensor_components!(
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
    q::Int,
    amp_plus::ComplexF64,
    amp_minus::ComplexF64,
    nᵢnⱼ::SVector{6,Float64},
)
    # Using a StaticArrays-friendly loop or just @inbounds
    @inbounds for I in 1:6
        A_plus[I, q]  = muladd(amp_plus,  nᵢnⱼ[I], A_plus[I, q])
        A_minus[I, q] = muladd(amp_minus, nᵢnⱼ[I], A_minus[I, q])
    end
    return nothing
end

# ============================================================
# Mode accumulation
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

function accumulate_marker_modes!(
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
    ks::AbstractVector{<:Real},
    τ_stop::Float64,
    n̂:: SVector{3,Float64},
    w::Float64,
    ws::SourceModeWorkspace,
    v::Float64,
)
    nᵢnⱼ = outer_prod(n̂)

    c_plus  =  1.0 - v*n̂[3]
    c_minus = -1.0 - v*n̂[3]

    phase_plus  = ws.phase_plus
    phase_minus = ws.phase_minus

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        I3_plus  = I3_zero_lower(k*c_plus,  τ_stop)
        I3_minus = I3_zero_lower(k*c_minus, τ_stop)

        amp_plus  = w * phase_plus[q]  * I3_plus
        amp_minus = w * phase_minus[q] * I3_minus

        accumulate_tensor_components!(
            A_plus, A_minus, q,
            amp_plus, amp_minus, nᵢnⱼ,
        )
    end

    return nothing
end

function accumulate_marker_modes!(
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
    ks::AbstractRange{<:Real},
    τ_stop::Float64,
    n̂:: SVector{3,Float64},
    w::Float64,
    ws::SourceModeWorkspace,
    v::Float64)

    nᵢnⱼ = outer_prod(n̂)

    c_plus  =  1.0 - v*n̂[3]
    c_minus = -1.0 - v*n̂[3]

    compute_sincos_grid!(
        ws.S_plus,
        ws.C_plus,
        ks,
        c_plus,
        τ_stop,
    )

    compute_sincos_grid!(
        ws.S_minus,
        ws.C_minus,
        ks,
        c_minus,
        τ_stop,
    )

    phase_plus  = ws.phase_plus
    phase_minus = ws.phase_minus

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        α_plus  = k * c_plus
        α_minus = k * c_minus

        I3_plus = I3_zero_lower_from_sincos(
            α_plus,
            τ_stop,
            ws.S_plus[q],
            ws.C_plus[q],
        )

        I3_minus = I3_zero_lower_from_sincos(
            α_minus,
            τ_stop,
            ws.S_minus[q],
            ws.C_minus[q],
        )

        amp_plus  = w * phase_plus[q]  * I3_plus
        amp_minus = w * phase_minus[q] * I3_minus

        accumulate_tensor_components!(
            A_plus, A_minus, q,
            amp_plus, amp_minus,
            nᵢnⱼ,
        )
    end

    return nothing
end