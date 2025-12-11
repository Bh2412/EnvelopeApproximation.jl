module QuadGKCFT

using QuadGK
using LinearAlgebra
using StaticArrays
import EnvelopeApproximation.CFTInterface: CFTPlan, fourier_modes

export VectorQuadGKPlan
export fourier_modes

"""
    VectorQuadGKPlan{K, T_Buf}

A plan managing the resources for QuadGK integration of a K-dimensional vector-valued function
against a set of Fourier modes.

# Type Parameters
- `K`: The dimension of the physics output (e.g., 2 for projected GWs, 6 for full tensor).
"""
mutable struct VectorQuadGKPlan{K} <: CFTPlan
    segbuf::Vector{<:QuadGK.Segment}  # Segment buffer for QuadGK
    phase_buffer::Vector{ComplexF64} # Internal temporary for broadcasting k values
    rtol::Float64
    atol::Float64
    initdiv:: Int  # The amount of splitting to do initially (in order to handle sharp features between nodes)
end

function VectorQuadGKPlan{K}(; rtol=1e-8, atol=1e-12, initdiv=1) where {K}
    return VectorQuadGKPlan{K}(alloc_segbuf(Float64, Matrix{ComplexF64}, Float64), ComplexF64[], rtol, atol, initdiv)
end

"""
    fourier_modes(f, plan, limits...)

Integrates `f(μ) * cis(-k * μ)` for all k in the plan simultaneously.

# Arguments
- `f`: Function taking `μ` and returning `SVector{K, Float64}` or `Vector{Float64}`.
- `scale`: Scaling factor for k (e.g., bubble radius).
- `limits`: Integration boundaries and discontinuities (e.g., -1.0, 1.0).
"""
function fourier_modes(f, ks:: AbstractVector{<: Real}, a:: Real, b:: Real, plan::VectorQuadGKPlan{K})::Tuple{Matrix{ComplexF64}, Float64} where {K}
    Nk = length(ks)

    # 1. Resize phase buffer if new ks vector is larger than previous runs
    if length(plan.phase_buffer) != Nk
        resize!(plan.phase_buffer, Nk)
    end

    # Define the vectorized integrand
    function integrand(μ)
        val = f(μ)# Expected to return SVector{K} or Vector{K}
        @inbounds @simd for i in 1:Nk
            plan.phase_buffer[i] = cis(-ks[i] * μ)
        end
        return val * transpose(plan.phase_buffer)
    end

    ps = range(a, b, plan.initdiv + 1)  

    val, err = quadgk(integrand, ps...; 
                      segbuf=plan.segbuf, 
                      rtol=plan.rtol, 
                      atol=plan.atol)
    
    return val, err
end

end