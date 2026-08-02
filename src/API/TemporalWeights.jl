using StaticArrays: SVector

abstract type TemporalWeight end
struct CosineWeight <: TemporalWeight end
struct ConstantWeight <: TemporalWeight end

"""
    DiracDelta(times)

Temporal weight that evaluates the stress tensor at each prescribed time in
`times`.
"""
struct DiracDelta{Times<:AbstractVector{<:Real}} <: TemporalWeight
    times::Times
end

"""
    ComplexExponential(ωs)

Temporal weight `exp(iωt)` evaluated at each frequency in the fixed vector
`ωs`. The number of temporal frequencies is encoded by the type parameter
`N` in `ComplexExponential{N}`.
"""
struct ComplexExponential{N} <: TemporalWeight
    ωs::SVector{N,Float64}
end

ComplexExponential(ωs::AbstractVector{<:Real}) =
    ComplexExponential(SVector{length(ωs),Float64}(ωs))

ComplexExponential(ωs::NTuple{N,<:Real}) where {N} =
    ComplexExponential(SVector{N,Float64}(ωs))

export TemporalWeight, CosineWeight, ConstantWeight, ComplexExponential,
       DiracDelta
