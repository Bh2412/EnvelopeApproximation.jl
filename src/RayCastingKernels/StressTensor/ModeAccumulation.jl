import Base.resize!

include("SinCosGrid.jl")

abstract type ModeWorkspace{W<:TemporalWeight} end

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
