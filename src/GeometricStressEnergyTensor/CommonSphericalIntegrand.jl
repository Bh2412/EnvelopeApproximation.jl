function _buffers(domes:: Vector{IntersectionDome}):: Tuple{Vector{PeriodicInterval}, Vector{Tuple{Float64, Float64}}, Vector{PeriodicInterval}}
    N = length(domes)
    arcs_buffer = Vector{PeriodicInterval}(undef, N)
    limits_buffer = Vector{Tuple{Float64, Float64}}(undef, 2N)
    intersection_buffer = Vector{PeriodicInterval}(undef, N)
    return arcs_buffer, limits_buffer, intersection_buffer
end

struct BubbleArcSurfaceIntegrand <: SphericalIntegrand{MVector{6, Float64}}
    R:: Float64
    domes:: Vector{IntersectionDome}
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}

    function BubbleArcSurfaceIntegrand(R:: Float64, domes:: Vector{IntersectionDome}):: BubbleArcSurfaceIntegrand
        return new(R, domes, _buffers(domes)...)
    end
end

function ∫_ϕ(basi:: BubbleArcSurfaceIntegrand, μ:: Float64):: MVector{6, Float64}
    V = zeros(MVector{6, Float64})
    intervals = ring_domes_intersection!(μ, basi.R, basi.domes, basi.arcs_buffer, 
                                         basi.limits_buffer, basi.intersection_buffer)
    for interval in intervals
        V .+= ∫_ϕ(upper_right, μ, interval.ϕ1, mod2π(interval.ϕ1 + interval.Δ))
    end
    return V
end

function *(rotation:: SMatrix{3, 3, Float64}, basi:: BubbleArcSurfaceIntegrand):: BubbleArcSurfaceIntegrand
    return BubbleArcSurfaceIntegrand(basi.R, (rotation, ) .* basi.domes)
end

const ZHat:: SphericalZhat = SphericalZhat()

struct BubbleArcPotentialIntegrand <: SphericalIntegrand{Float64}
    R:: Float64
    domes:: Vector{IntersectionDome}
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}

    function BubbleArcPotentialIntegrand(R:: Float64, domes:: Vector{IntersectionDome})
        return new(R, domes, _buffers(domes)...)
    end
end

function ∫_ϕ(bapi:: BubbleArcPotentialIntegrand, μ:: Float64):: Float64
    intervals = ring_domes_intersection!(μ, bapi.R, bapi.domes, 
                                         bapi.arcs_buffer, bapi.limits_buffer, bapi.intersection_buffer)
    return sum((∫_ϕ(ZHat, μ, interval.ϕ1, mod2π(interval.ϕ1 + interval.Δ)) for interval in intervals), init=0.)
end

const ZZ:: TensorDirection = SphericalZZ()
const Trace:: TensorDirection = SphericalTrace()

struct BubbleArck̂ik̂j∂iφ∂jφ <: SphericalIntegrand{Float64}
    R:: Float64
    domes:: Vector{IntersectionDome}
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}

    function BubbleArck̂ik̂j∂iφ∂jφ(R:: Float64, domes:: Vector{IntersectionDome})
        return new(R, domes, _buffers(domes)...)
    end
end

function ∫_ϕ(ba:: BubbleArck̂ik̂j∂iφ∂jφ, μ:: Float64):: Float64
    intervals = ring_domes_intersection!(μ, ba.R, ba.domes, 
                                         ba.arcs_buffer, ba.limits_buffer, ba.intersection_buffer)
    return sum((∫_ϕ(ZZ, μ, interval.ϕ1, mod2π(interval.ϕ1 + interval.Δ)) for interval in intervals), init=0.)
end

struct Ŋ <: SphericalIntegrand{Float64} end

∫_ϕ(st:: Ŋ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (μ ^ 2 - 1. / 3) * (ϕ2 - ϕ1)

const ŋ:: SphericalIntegrand{Float64} = Ŋ()

#  equivalent to a concatenation by k̂ik̂j - 1. / 3 ⋅ δij

struct BubbleArcŊ <: SphericalIntegrand{Float64}
    R:: Float64
    domes:: Vector{IntersectionDome}
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}

    function BubbleArcŊ(R:: Float64, domes:: Vector{IntersectionDome})
        return new(R, domes, _buffers(domes)...)
    end
end

function ∫_ϕ(ba:: BubbleArcŊ, μ:: Float64):: Float64
    intervals = ring_domes_intersection!(μ, ba.R, ba.domes, 
                                         ba.arcs_buffer, ba.limits_buffer, ba.intersection_buffer)
    return sum((∫_ϕ(ŋ, μ, interval.ϕ1, mod2π(interval.ϕ1 + interval.Δ)) for interval in intervals), init=0.)
end


