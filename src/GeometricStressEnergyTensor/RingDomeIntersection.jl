

const lib_path = "$(dirname(dirname(pathof(EnvelopeApproximation))))/rust_extensions/benrust/obj/fastrust.so"

struct Vector3
    x:: Float64
    y:: Float64
    z:: Float64
end

function Vector3(v:: Vec3):: Vector3
    return Vector3(v[1], v[2], v[3])
end

ring_dome_intersection(μ:: Float64, R:: Float64, n̂:: Vector3, h:: Float64, dome_like:: Bool):: PeriodicInterval = @ccall lib_path.ring_dome_intersect(μ:: Float64, R:: Float64, n̂:: Vector3, h:: Float64, dome_like:: Bool):: PeriodicInterval

# This function returns the intersection between the integratoin ring and the dome like region
# of the intersection of 2 bubbles
function ring_dome_intersection(μ′:: Float64, R:: Float64, n̂′:: Vec3, h:: Float64, dome_like:: Bool):: PeriodicInterval
    return ring_dome_intersection(μ′, R, Vector3(n̂′), h, dome_like)
end

# A prime indicates that the intersection is in a rotated coordinate system
function ring_dome_intersection(μ′:: Float64, R:: Float64, 
                                intersection′:: IntersectionArc):: PeriodicInterval
    n̂′, h, dome_like = intersection′.n̂, intersection′.h, intersection′.dome_like
    return ring_dome_intersection(μ′, R, n̂′, h, dome_like) 
end
