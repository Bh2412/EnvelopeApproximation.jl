
const ZZ:: TensorDirection = SphericalZZ()
const Trace:: TensorDirection = SphericalTrace()

struct BubbleArck̂ik̂j∂iφ∂jφ <: SphericalIntegrand{Float64}
    R:: Float64
    arcs:: Vector{IntersectionArc}
end

function ∫_ϕ(ba:: BubbleArck̂ik̂j∂iφ∂jφ, μ:: Float64):: Float64
    intervals = Δϕ′(μ, ba.R, ba.arcs).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ZZ, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end

function bubble_k̂ik̂j∂iφ∂jφ_contribution(k:: Vec3, bubble:: Bubble, 
                                        arcs:: Vector{IntersectionArc}, 
                                        krotation:: SMatrix{3, 3, Float64}, 
                                        ΔV:: Float64; kwargs...):: ComplexF64
    # Rotate to a coordinate system in which k̂ik̂j = δi3δj3
    mode = fourier_mode(BubbleArck̂ik̂j∂iφ∂jφ(bubble.radius, (krotation, ) .* arcs), bubble.radius * norm(k); kwargs...)
    return mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function k̂ik̂jTij(k:: Vec3, bubbles:: Bubbles, 
                 arcs:: Dict{Int64, Vector{IntersectionArc}},
                 krotation:: SMatrix{3, 3, Float64}, 
                 ΔV:: Float64; kwargs...):: ComplexF64
    V = 0.
    for (bubble_index, bubble_arcs) in arcs
        V += bubble_k̂ik̂j∂iφ∂jφ_contribution(k, bubbles[bubble_index], 
                                            bubble_arcs, krotation, ΔV; kwargs...)
        V -= bubble_potential_contribution(k, bubbles[bubble_index], 
                                           bubble_arcs, krotation, ΔV; kwargs...)
    end
    return V
end

function k̂ik̂jTij(ks:: Vector{Vec3}, bubbles:: Bubbles;
                 arcs:: Union{Nothing, Dict{Int64, Vector{IntersectionArc}}} = nothing, 
                 krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                 ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    arcs ≡ nothing && (arcs = intersection_arcs(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return @. k̂ik̂jTij(ks, (bubbles, ), (arcs, ), krotations, (ΔV, ); kwargs...)
end

export k̂ik̂jTij

struct Ŋ <: SphericalIntegrand{Float64} end

∫_ϕ(st:: Ŋ, μ:: Float64, ϕ1:: Float64, ϕ2:: Float64):: Float64 = (μ ^ 2 - 1. / 3) * (ϕ2 - ϕ1)

const ŋ:: SphericalIntegrand{Float64} = Ŋ()

#  equivalent to a concatenation by k̂ik̂j - 1. / 3 ⋅ δij

struct BubbleArcŊ <: SphericalIntegrand{Float64}
    R:: Float64
    arcs:: Vector{IntersectionArc}
end

function ∫_ϕ(ba:: BubbleArcŊ, μ:: Float64):: Float64
    intervals = Δϕ′(μ, ba.R, ba.arcs).items
    _f(i:: Interval{Float64, Closed, Closed}):: Float64 = ∫_ϕ(ŋ, μ, i.first, i.last)
    x:: Float64 = 0.
    for i in intervals
        x += _f(i)
    end
    return x
end

function bubble_Ŋ_contribution(k:: Vec3, bubble:: Bubble, 
                               arcs:: Vector{IntersectionArc}, 
                               krotation:: SMatrix{3, 3, Float64}, 
                               ΔV:: Float64; kwargs...):: ComplexF64
    mode = fourier_mode(BubbleArcŊ(bubble.radius, (krotation, ) .* arcs), bubble.radius * norm(k); kwargs...)
    return mode * ((ΔV * (bubble.radius ^ 3) / 3) * cis(-(k ⋅ bubble.center.coordinates)))
end

function ŋ_source(k:: Vec3, bubbles:: Bubbles, 
                  arcs:: Dict{Int64, Vector{IntersectionArc}},
                  krotation:: SMatrix{3, 3, Float64}, 
                  ΔV:: Float64; kwargs...):: ComplexF64
    V = 0.
    for (bubble_index, bubble_arcs) in arcs
        V += bubble_Ŋ_contribution(k, bubbles[bubble_index], 
                                   bubble_arcs, krotation, ΔV; kwargs...)
    end
    return V
end

function ŋ_source(ks:: Vector{Vec3}, bubbles:: Bubbles;
                  arcs:: Union{Nothing, Dict{Int64, Vector{IntersectionArc}}} = nothing, 
                  krotations:: Union{Nothing, Vector{<: SMatrix{3, 3, Float64}}} = nothing, 
                  ΔV:: Float64 = 1., kwargs...):: Vector{ComplexF64}
    arcs ≡ nothing && (arcs = intersection_arcs(bubbles))
    krotations ≡ nothing && (krotations = align_ẑ.(ks))
    return ŋ_source.(ks, (bubbles, ), (arcs, ), krotations, (ΔV, ); kwargs...)
end

export ŋ_source
