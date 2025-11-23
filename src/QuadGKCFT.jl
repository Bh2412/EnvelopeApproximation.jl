module QuadGKCFT

using QuadGK
using LinearAlgebra
using StaticArrays

export VectorQuadGKPlan
export fourier_modes

"""
    VectorQuadGKPlan{K, T_Buf}

A plan managing the resources for QuadGK integration of a K-dimensional vector-valued function
against a set of Fourier modes.

# Type Parameters
- `K`: The dimension of the physics output (e.g., 2 for projected GWs, 6 for full tensor).
- `T_Buf`: The type of the internal segment buffer (inferred automatically).
"""
mutable struct VectorQuadGKPlan{K}
    segbuf::Vector{<:QuadGK.Segment}  # Segment buffer for QuadGK
    phase_buffer::Vector{ComplexF64} # Internal temporary for broadcasting k values
    rtol::Float64
    atol::Float64
end

function VectorQuadGKPlan{K}(; rtol=1e-8, atol=1e-12) where {K}
    return VectorQuadGKPlan{K}(alloc_segbuf(Float64, Matrix{ComplexF64}, Float64), ComplexF64[], rtol, atol)
end

"""
    fourier_modes(f, plan, limits...)

Integrates `f(μ) * cis(-k * μ)` for all k in the plan simultaneously.

# Arguments
- `f`: Function taking `μ` and returning `SVector{K, Float64}` or `Vector{Float64}`.
- `scale`: Scaling factor for k (e.g., bubble radius).
- `limits`: Integration boundaries and discontinuities (e.g., -1.0, 1.0).
"""
function fourier_modes(f, ks:: Vector{<: Real}, plan::VectorQuadGKPlan{K}, limits::Real..., )::Matrix{ComplexF64} where {K}
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

    val, err = quadgk(integrand, limits...; 
                      segbuf=plan.segbuf, 
                      rtol=plan.rtol, 
                      atol=plan.atol)
    
    return transpose(val) 
end

end