function intersecting(bubble1:: Bubble, bubble2:: Bubble):: Bool
    return euc(bubble1.center, bubble2.center) < bubble1.radius + bubble2.radius
end

struct IntersectionDome
    h:: Float64
    n̂:: Vec3
    dome_like:: Bool
end

function IntersectionDome(n:: Vec3, dome_like:: Bool)
    h = norm(n)
    return IntersectionDome(h, n / h, dome_like)
end

function *(rotation:: SMatrix{3, 3, Float64}, arc:: IntersectionDome):: IntersectionDome
    return IntersectionDome(arc.h, rotation * arc.n̂, arc.dome_like)
end


export IntersectionDome

#=
Credit to WolFram Malthworld sphere-sphere intersection article
=#
function λ(r1:: Float64, r2:: Float64, d:: Float64) 
    return (d^2 + r1^2 - r2^2) / 2d
end

function ∩(bubble1:: Bubble, bubble2:: Bubble):: Tuple{IntersectionDome, IntersectionDome}
    n = bubble2.center - bubble1.center
    d = norm(n)
    n̂ = n / d 
    _λ = λ(bubble1.radius, bubble2.radius, d)
    n1 = _λ * n̂
    n2 = -n + n1
    domelike1 = _λ > 0.
    domelike2 = λ(bubble2.radius, bubble1.radius, d) > 0.
    return (IntersectionDome(n1, domelike1), IntersectionDome(n2, domelike2))
end

function intersection_domes(bubbles:: Bubbles):: Dict{Int, Vector{IntersectionDome}}
    d = Dict{Int, Vector{IntersectionDome}}()
    for i in eachindex(bubbles)
        d[i] = Vector{IntersectionDome}()
    end
    for (i, bubble1) in enumerate(bubbles)
        for (j̃, bubble2) in enumerate(bubbles[(i + 1):end])
            j = j̃ + i
            if intersecting(bubble1, bubble2)
                intersection1, intersection2 = bubble1 ∩ bubble2
                for (k, intersection) in ((i, intersection1), (j, intersection2))
                    push!(d[k], intersection)
                end
            else
                continue
            end        
        end
    end
    return d
end

export intersection_domes
