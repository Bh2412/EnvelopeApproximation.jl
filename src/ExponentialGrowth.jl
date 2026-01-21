
"""
    appropriate_t_end(tvp:: Float64, β:: Float64, Γ_0:: Float64; t_0:: Float64 = 0.):: Float64
Given a target true vacuum fraction `tvp` in (0, 1), the nucleation rate parameters `β` and `Γ_0`,
and an optional `t_0` (the time in which the nucleation rate is Γ_0, default 0.), compute the average completion time of the PT
"""
function appropriate_t_end(tvp:: Float64, β:: Float64, Γ_0:: Float64, v_wall:: Float64; t_0:: Float64 = 0.):: Float64
    return t_0 + (1 / β) * log(-β ^ 4 * log(1. - tvp) / (8π * v_wall^3 * Γ_0))
end

struct ExponentialGrowthProcess <: NucleationProcess
    β:: Float64
    Γ_0:: Float64
    t_0:: Float64
    tvp:: Float64
    v_wall:: Float64
    t_end:: Float64

    function ExponentialGrowthProcess(β:: Float64, Γ_0:: Float64, 
                                      tvp:: Float64; t_0:: Float64 = 0., 
                                      v_wall:: Float64 = 1.)
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
        return new(β, Γ_0, t_0, tvp, v_wall, t_end)
    end
end

function Λ(egp:: ExponentialGrowthProcess, t1:: Float64, t2:: Float64):: Float64
    β = egp.β
    Γ_0 = egp.Γ_0
    t_0 = egp.t_0
    return Γ_0 * (1 / β) * (exp(β * (t2 - t_0)) - exp(β * (t1 - t_0)))
end

"""
    N_expected_value(egp:: ExponentialGrowthProcess, space:: AbstractSpace):: Float64
Compute the expected number of *total* nucleations in the given space for the ExponentialGrowthProcess `egp`.
This includes non-physical nucleations that may be filtered out later.
"""
function N_expected_value(egp:: ExponentialGrowthProcess, space:: AbstractSpace):: Float64
    return Λ(egp, -Inf, egp.t_end) * volume(space)
end

function sample_nucleation_times(rng:: AbstractRNG, process:: ExponentialGrowthProcess, N:: Int64):: Vector{Float64}
    us = rand(rng, N)
    return process.t_end .+ (1 / process.β) * log.(1 .- us)
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

function sample_nucleations(rng:: AbstractRNG, process:: ExponentialGrowthProcess, 
                            space:: AbstractSpace, boundary_condition:: Periodic; padding:: Float64 = 0.)::Vector{Nucleation}
    bbox = padded_box(space, padding)
    N = rand(rng, Poisson(N_expected_value(process, bbox)))
    nucleations = sample(rng, N, bbox)
    nucleation_times = sample_nucleation_times(rng, process, N)
    sort!(nucleation_times)
    tv_nucleations = Vector{Nucleation}()
    sizehint!(tv_nucleations, N)
    wall_radius = radial_profile(process)
    for (t, p) in zip(nucleation_times, nucleations)
        phys_nucleation = true
        for nucleation in tv_nucleations
            if toroidal_dist(p, nucleation.site, bbox.L) <= wall_radius(t - nucleation.time)
                phys_nucleation = false
                break
            end
        end
        if phys_nucleation
            push!(tv_nucleations, Nucleation((time=t, site=p)))
        end
    end    
    return filter!(n -> n.site ∈ space, tv_nucleations)
end
