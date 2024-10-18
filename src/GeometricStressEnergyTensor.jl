module GeometricStressEnergyTensor

using EnvelopeApproximation.BubbleBasics
using LinearAlgebra

function intersecting(bubble1:: Bubble, bubble2:: Bubble):: Bool
    return euc(bubble1.center - bubble2.center) < bubble1.radius + bubble2.radius
end

struct Intersection
    h:: Float64
    n̂:: Vec3
    includes_center:: Bool
end

function Intersection(n:: Vec3, includes_center:: Bool)
    h = norm(n)
    return Intersection(h, n / h, includes_center)
end

#=
Credit to WolFram Malthworld sphere-sphere intersection article
=#
function λ(r1:: Float64, r2:: Float64, d:: Float64) 
    x = (d^2 + r1 ^2 - r2 ^ 2) / 2d
    return x
end

function ∩(bubble1:: Bubble, bubble2:: Bubble):: Tuple{Intersection, Interection}
    n = bubble2.center - bubble1.center
    d = norm(n)
    _λ = λ(bubble1.radius, bubble2.radius, d)
    n1 = _λ * n
    n2 = -n + n1
    in1 = bubble1.center ∈ bubble2
    in2 = bubble2.center ∈ bubble1
    return (Intersection(n1, in1), Intersection(n2, in2))
end

function intersections(bubbles:: Bubbles):: Dict{Int, Vector{Intersection}}
    d = Dict{Int, Vector{Intersection}}()
    for (i, bubble1) in enumerate(bubbles)
        for (j̃, bubble2) in bubbles[(i + 1):end]
            j = j̃ + i
            if intersecting(bubble1, bubble2)
                intersection1, intersection2 = bubble1 ∩ bubble2
                for (k, intersection) in ((i, intersection1), (j, intersection2))
                    if k ∈ keys(d)
                        push!(d[k], intersection)
                    else
                        d[k] = Vector{Intersection}([intersection])
                    end
                end
            else
                continue
            end        
        end
    end
    return d
end



end