
"""
    appropriate_t_end(tvp:: Float64, β:: Float64, Γ_0:: Float64; t_0:: Float64 = 0.):: Float64
Given a target true vacuum fraction `tvp` in (0, 1), the nucleation rate parameters `β` and `Γ_0`,
and an optional `t_0` (the time in which the nucleation rate is Γ_0, default 0.), compute the average completion time of the PT
"""
function appropriate_t_end(tvp:: Float64, β:: Float64, Γ_0:: Float64, v_wall:: Float64; t_0:: Float64 = 0.):: Float64
    return t_0 + (1 / β) * log(-β ^ 4 * log(1. - tvp) / (8π * v_wall^3 * Γ_0))
end

abstract type ExponentialGrowthSamplingMethod end
struct SequentialSampling <: ExponentialGrowthSamplingMethod end
struct APrioriPoissonSampling <: ExponentialGrowthSamplingMethod end

struct ExponentialGrowthProcess{T <: ExponentialGrowthSamplingMethod} <: NucleationProcess
    β:: Float64
    Γ_0:: Float64
    t_0:: Float64
    tvp:: Float64
    v_wall:: Float64
    t_end:: Float64

    function ExponentialGrowthProcess{T}(β:: Float64, Γ_0:: Float64, 
                                         tvp:: Float64; t_0:: Float64 = 0., 
                                         v_wall:: Float64 = 1.) where {T <: ExponentialGrowthSamplingMethod}
        if β <= 0
            throw(ArgumentError("β must be positive"))
        end
        if Γ_0 <= 0
            throw(ArgumentError("Γ_0 must be positive"))
        end
        if tvp <= 0 || tvp >= 1
            throw(ArgumentError("tvp must be in (0, 1)"))
        end
        if v_wall <= 0 || v_wall > 1
            throw(ArgumentError("v_wall must be in (0, 1]"))
        end
        t_end = appropriate_t_end(tvp, β, Γ_0, v_wall; t_0=t_0)
        return new{T}(β, Γ_0, t_0, tvp, v_wall, t_end)
    end
end

function ExponentialGrowthProcess(β:: Float64, Γ_0:: Float64, 
                                  tvp:: Float64; t_0:: Float64 = 0., 
                                  v_wall:: Float64 = 1.)
    return ExponentialGrowthProcess{APrioriPoissonSampling}(β, Γ_0, tvp; t_0=t_0, v_wall=v_wall)
end

function Λ(t1:: Float64, t2:: Float64, β:: Float64, Γ_0:: Float64, t_0:: Float64):: Float64
    return Γ_0 * (1 / β) * (exp(β * (t2 - t_0)) - exp(β * (t1 - t_0)))
end

function Λ(egp:: ExponentialGrowthProcess, t1:: Float64, t2:: Float64):: Float64
    return Λ(t1, t2, egp.β, egp.Γ_0, egp.t_0)
end

function expected_candidate_nucleations(t_start:: Float64, t_end:: Float64, V:: Float64, β:: Float64, Γ_0:: Float64, t_0:: Float64):: Float64
    return Λ(t_start, t_end, β, Γ_0, t_0) * V
end

"""
    expected_candidate_nucleations(egp:: ExponentialGrowthProcess, space:: AbstractSpace):: Float64
Compute the expected number of *total* nucleations in the given space for the ExponentialGrowthProcess `egp`.
This includes non-physical nucleations that may be filtered out later.
"""
function expected_candidate_nucleations(egp:: ExponentialGrowthProcess, space:: AbstractSpace):: Float64
    return Λ(egp, -Inf, egp.t_end) * volume(space)
end

function sample_backwards_exponential(rng:: AbstractRNG, N:: Int64, t_end:: Float64, β:: Float64):: Vector{Float64}
    us = rand(rng, N)
    return @. t_end + (1 / β) * log(1 - us)
end

function sample_backwards_exponential(rng:: AbstractRNG, process:: ExponentialGrowthProcess, N:: Int64):: Vector{Float64}
    return sample_backwards_exponential(rng, N, process.t_end, process.β)
end

function next_event_time(rng:: AbstractRNG, t_last:: Float64, V:: Float64, β:: Float64, Γ_0:: Float64, t_0:: Float64):: Float64
    u = rand(rng)
    prev_term = t_last == -Inf ? 0.0 : exp(β * (t_last - t_0))
    return t_0 + (1 / β) * log(prev_term - (β / (V * Γ_0)) * log(1 - u))
end

function next_event_time(rng:: AbstractRNG, t_last:: Float64, V:: Float64, process:: ExponentialGrowthProcess):: Float64
    return next_event_time(rng, t_last, V, process.β, process.Γ_0, process.t_0)
end

function radial_profile(process:: ExponentialGrowthProcess):: Function
    v_wall = process.v_wall
    return t -> v_wall * t
end

function completion_time(process:: ExponentialGrowthProcess):: Float64
    return process.t_end
end

function toroidal_dist(p1::Point3, p2::Point3, L::Float64)::Float64
    # Calculate difference in each dimension
    c1 = coordinates(p1)
    c2 = coordinates(p2)
    dx = abs(c1[1] - c2[1])
    dy = abs(c1[2] - c2[2])
    dz = abs(c1[3] - c2[3])

    # Wrap: if dist > L/2, taking the path across the boundary is shorter
    if dx > L/2; dx = L - dx; end
    if dy > L/2; dy = L - dy; end
    if dz > L/2; dz = L - dz; end
    
    return sqrt(dx^2 + dy^2 + dz^2)
end

function padded_box(space:: AbstractSpace, padding:: Float64):: BoxSpace
    bbox = bounding_box(space)
    return BoxSpace(bbox.L + 2 * padding, bbox.center)
end

function sample_candidate_nucleations(rng:: AbstractRNG, process:: ExponentialGrowthProcess{APrioriPoissonSampling}, space:: AbstractSpace)
    N = rand(rng, Poisson(expected_candidate_nucleations(process, space)))
    nucleation_sites = sample(rng, N, space)
    nucleation_times = sample_backwards_exponential(rng, process, N)
    sort!(nucleation_times)
    return zip(nucleation_times, nucleation_sites)
end

function is_physical(t:: Float64, p:: Point3, L:: Float64, tv_nucleations:: Vector{Nucleation}, wall_radius)
    phys_nucleation = true
    for nucleation in tv_nucleations
        if toroidal_dist(p, nucleation.site, L) <= wall_radius(t - nucleation.time)
            phys_nucleation = false
            break
        end
    end
    return phys_nucleation
end

function sample_nucleations(rng:: AbstractRNG, process:: ExponentialGrowthProcess{APrioriPoissonSampling}, 
                            space:: AbstractSpace, boundary_condition:: Periodic; padding:: Float64 = 0.)::Vector{Nucleation}
    bbox = padded_box(space, padding)
    candidate_nucleations = sample_candidate_nucleations(rng, process, bbox)
    tv_nucleations = Vector{Nucleation}()
    sizehint!(tv_nucleations, length(candidate_nucleations))
    L = bbox.L
    wall_radius = radial_profile(process)
    for (t, p) in candidate_nucleations
        if is_physical(t, p, L, tv_nucleations, wall_radius)
            push!(tv_nucleations, Nucleation((time=t, site=p)))
        end
    end    
    return filter!(n -> n.site ∈ space, tv_nucleations)
end

function sample_nucleations(rng:: AbstractRNG, process:: ExponentialGrowthProcess{SequentialSampling}, 
                            space:: AbstractSpace, boundary_condition:: Periodic; padding:: Float64 = 0.)::Vector{Nucleation}
    bbox = padded_box(space, padding)
    V = volume(bbox)
    t_last = -Inf
    
    tv_nucleations = Vector{Nucleation}()
    wall_radius = radial_profile(process)
    L = bbox.L
    
    β = process.β
    Γ_0 = process.Γ_0
    t_0 = process.t_0
    
    while true
        t_cand = next_event_time(rng, t_last, V, β, Γ_0, t_0)
        
        if t_cand > process.t_end
            break
        end
        
        p = sample(rng, bbox)
        
        if is_physical(t_cand, p, L, tv_nucleations, wall_radius)
            push!(tv_nucleations, Nucleation((time=t_cand, site=p)))
        end
        
        t_last = t_cand
    end
    
    return filter!(n -> n.site ∈ space, tv_nucleations)
end
