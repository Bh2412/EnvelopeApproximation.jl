struct BubbleArcSurfaceIntegrand <: SphericalIntegrand{MVector{6, Float64}}
    R:: Float64
    domes:: Vector{IntersectionDome}
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}

    function BubbleArcSurfaceIntegrand(R:: Float64, domes:: Vector{IntersectionDome}):: BubbleArcSurfaceIntegrand
        N = length(domes)
        arcs_buffer = Vector{PeriodicInterval}(under, N)
        limits_buffer = Vector{Tuple{Float64, Float64}}(undef, 2N)
        intersection_buffer = Vector{PeriodicInterval}(undef, N)
        return new(R, domes, arcs_buffer, limits_buffer, intersection_buffer)
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
end

function ∫_ϕ(bapi:: BubbleArcPotentialIntegrand, μ:: Float64):: Float64
    intervals = Δϕ′(μ, bapi.R, bapi.domes).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ZHat, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end

const ZZ:: TensorDirection = SphericalZZ()
const Trace:: TensorDirection = SphericalTrace()

struct BubbleArck̂ik̂j∂iφ∂jφ <: SphericalIntegrand{Float64}
    R:: Float64
    domes:: Vector{IntersectionDome}
end

function ∫_ϕ(ba:: BubbleArck̂ik̂j∂iφ∂jφ, μ:: Float64):: Float64
    intervals = Δϕ′(μ, ba.R, ba.domes).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ZZ, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end

struct Ŋ <: SphericalIntegrand{Float64} end

∫_ϕ(st:: Ŋ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (μ ^ 2 - 1. / 3) * (ϕ2 - ϕ1)

const ŋ:: SphericalIntegrand{Float64} = Ŋ()

#  equivalent to a concatenation by k̂ik̂j - 1. / 3 ⋅ δij

struct BubbleArcŊ <: SphericalIntegrand{Float64}
    R:: Float64
    domes:: Vector{IntersectionDome}
end

function ∫_ϕ(ba:: BubbleArcŊ, μ:: Float64):: Float64
    intervals = Δϕ′(μ, ba.R, ba.domes).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ŋ, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end


