"""
Pads the simulation domain with periodic 'ghost' bubbles out to the maximum 
distance ANY ray could travel.
"""
function periodic_nucleations(original_nucleations:: Vector{Nucleation}, 
                              v::Float64, t_start::Float64, t_end::Float64, box::BoxSpace):: Vector{Nucleation}
    original_times = map(original_nucleations) do nuc
    nuc[:time]
    end

    original_centers = map(original_nucleations) do nuc
        nuc[:site].coordinates
    end

    bubbles = Bubble[]
    sizehint!(bubbles, length(original_centers))
    
    # The maximum distance ANY ray can travel outward from the primary box
    D_ray_max = v * (t_end - t_start)
    
    for i in eachindex(original_centers)
        # The maximum distance this specific bubble can expand
        R_b_max = v * (t_end - original_times[i])
        
        # A ghost is needed if a ray can reach its expanding wall
        R_pad = D_ray_max + R_b_max
        
        push!(bubbles, Bubble(Point3(original_centers[i]), R_pad))
    end

    # Generate periodic copies using a optimized PBC tool for periodic bubble truncation.
    origin_map = Int[]
    padded_bubbles = append_periodic_bubbles!(bubbles, box, origin_map)

    # Reconstruct centers and times for the fully padded domain.
    padded_centers = [b.center for b in padded_bubbles]
    padded_times = original_times[origin_map] 

    return map((t, s) -> (time=t, site=s), padded_times, padded_centers)
end