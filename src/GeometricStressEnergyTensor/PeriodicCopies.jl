const box_faces:: Vector{SVector{3, Int}} = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
const box_edges:: Vector{SVector{3, Int}} = [
    (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), # Z-parallel edges
    (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1), # Y-parallel edges
    (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1)  # X-parallel edges
]
const box_vertices:: Vector{SVector{3, Int}} = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
]
# Map Edges -> Indices of the 2 faces that form them
const edge_face_correspondence:: Vector{SVector{2, Int}} = [
    (1, 3), (1, 4), (2, 3), (2, 4),
    (1, 5), (1, 6), (2, 5), (2, 6),
    (3, 5), (3, 6), (4, 5), (4, 6)
]
# Map Vertices -> Indices of the 3 faces that form them
const vertex_face_correspondence:: Vector{SVector{3, Int}} = [
    (1, 3, 5), (1, 3, 6), (1, 4, 5), (1, 4, 6),
    (2, 3, 5), (2, 3, 6), (2, 4, 5), (2, 4, 6)
]

function face_distance(p:: Point3, face:: SVector{3, Int}, L_half:: Float64):: Float64
    axis = findfirst(!iszero, face)
    return abs(L_half * face[axis] - p.coordinates[axis])
end

function edge_distance(p:: Point3, edge:: SVector{3, Int}, L_half:: Float64):: Float64
    dist_sq = 0.0
    
    for i in 1:3
        if edge[i] != 0
            dist_sq += abs2(p.coordinates[i] - (edge[i] * L_half))
        end
    end
    return √(dist_sq)
end

function vertex_distance(p:: Point3, vertex:: SVector{3, Int}, L_half:: Float64):: Float64
    dist_sq = 0.0

    for i in 1:3
        dist_sq += abs2(p.coordinates[i] - (vertex[i] * L_half))
    end

    return √(dist_sq)
end

function periodic_copies!(periodic_bubbles:: Bubbles, 
                          bubble:: Bubble, box:: BoxSpace)
    # Implementation for generating periodic copies of bubbles
    intersecting_faces = zeros(Bool, 6)
    L_half = box.L / 2.0
    p = with_origin(bubble.center, box.center)
    for (i, face) in enumerate(box_faces)
        dist = face_distance(p, face, L_half)
        if dist < bubble.radius
            intersecting_faces[i] = true
            shift_vec = Vec3(face * box.L)
            push!(periodic_bubbles, Bubble(bubble.center - shift_vec, bubble.radius))
        end
    end
    for (i, edge) in enumerate(box_edges)
        face_indices = edge_face_correspondence[i]
        if all(view(intersecting_faces, face_indices))
            dist = edge_distance(p, edge, L_half)
            if dist < bubble.radius
                shift_vec = Vec3(edge * box.L)
                push!(periodic_bubbles, Bubble(bubble.center - shift_vec, bubble.radius))
            end
        end
    end
    for (i, vertex) in enumerate(box_vertices)
        face_indices = vertex_face_correspondence[i]
        if all(view(intersecting_faces, face_indices))
            dist = vertex_distance(p, vertex, L_half)
            if dist < bubble.radius
                shift_vec = Vec3(vertex * box.L)
                push!(periodic_bubbles, Bubble(bubble.center - shift_vec, bubble.radius))
            end
        end
    end
end

function unfold_periodic_bubbles(bubbles:: Bubbles, box:: BoxSpace):: Bubbles
    periodic_bubbles = Bubble[]
    sizehint!(periodic_bubbles, length(bubbles) * 26)  # Worst case scenario: each bubble spawns 26 copies
    for bubble in bubbles
        periodic_copies!(periodic_bubbles, bubble, box)
    end
    return periodic_bubbles
end
