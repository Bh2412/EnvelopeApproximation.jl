using StaticArrays
using LinearAlgebra

struct NucleationSoA
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    t::Vector{Float64}
end

function NucleationSoA(nucleations)
    N = length(nucleations)
    x = Vector{Float64}(undef, N)
    y = Vector{Float64}(undef, N)
    z = Vector{Float64}(undef, N)
    t = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        c = nucleations[j][:site].coordinates
        x[j] = c[1]
        y[j] = c[2]
        z[j] = c[3]
        t[j] = nucleations[j][:time]
    end
    return NucleationSoA(x, y, z, t)
end

mutable struct SourceCollisionWorkspace
    dx::Vector{Float64}
    dy::Vector{Float64}
    dz::Vector{Float64}
    dt::Vector{Float64}
    numer::Vector{Float64}
    tau_floor::Vector{Float64}
    candidates::Vector{Int}
end

function SourceCollisionWorkspace(N::Int)
    return SourceCollisionWorkspace(
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        sizehint!(Int[], N),
    )
end

"""
    collision_time(center_i::SVector{3, Float64}, center_j::SVector{3, Float64}, n̂::SVector{3, Float64},
                   t_i::Float64, t_j::Float64, v::Float64)::Union{Float64, Nothing}

Compute the collision time τ of a ray from bubble i along direction n̂ with bubble j.

The ray position at time τ (measured from t_i) is:
  x(τ) = x_i + v·τ·n̂

The ray collides with bubble j when:
  |x_i + v·τ·n̂ - x_j| = v·(t_i + τ - t_j)

Squaring and simplifying yields:
  τ = (v²Δt² - |Δx|²) / (2v(n̂ · Δx - v·Δt))

where Δx = x_i - x_j, Δt = t_i - t_j.

Returns τ if collision occurs (τ ≥ 0 and within time window); otherwise returns nothing.

# Arguments
- `center_i::SVector{3, Float64}`: Center of the source bubble
- `center_j::SVector{3, Float64}`: Center of the target bubble
- `n̂::SVector{3, Float64}`: Ray direction (unit vector)
- `t_i::Float64`: Nucleation time of bubble i
- `t_j::Float64`: Nucleation time of bubble j
- `v::Float64`: Bubble wall velocity

# Returns
Union{Float64, Nothing}: Collision time τ if valid, nothing otherwise
"""
function collision_time(center_i::SVector{3, Float64}, center_j::SVector{3, Float64}, n̂::SVector{3, Float64},
                        t_i::Float64, t_j::Float64, v::Float64)::Union{Float64, Nothing}
    # Compute Δx = x_i - x_j and Δt = t_i - t_j
    Δx = center_i - center_j
    Δt = t_i - t_j

    # Compute denominator: 2v(n̂·Δx - v·Δt)
    n̂_dot_Δx = dot(n̂, Δx)
    denom = 2.0 * v * (n̂_dot_Δx - v * Δt)

    # OPTIMIZATION: Since nucleations never overlap, numer (v²Δt² - |Δx|²) is ALWAYS <= 0.
    # Therefore, for τ to be positive, denom MUST be strictly negative.
    # If denom >= 0, the ray is moving away from the bubble or parallel, and will never collide.
    if denom > -1.0e-12
        return nothing
    end

    # Compute numerator: v²Δt² - |Δx|²
    Δx_norm_sq = sum(abs2, Δx)
    numer = v^2 * Δt^2 - Δx_norm_sq

    # Solve for τ
    τ = numer / denom

    # τ must be non-negative AND must occur after bubble j has nucleated.
    # The earliest valid τ is max(0, t_j − t_i) = max(0, −Δt).
    τ_min = max(0.0, -Δt)
    if τ < τ_min - 1.0e-12
        return nothing
    end

    return max(τ_min, τ)
end

"""
    find_collision_time(i::Int, center_i::SVector{3, Float64}, n̂::SVector{3, Float64},
                        nucleations::Vector, t_i::Float64, t_end::Float64, v::Float64)::Float64

Find the earliest collision time for a ray from bubble `i` along direction n̂.

Loops through all other bubbles (and ghosts) to compute their collision times,
maintaining a running minimum.

# Arguments
- `i::Int`: Index of the source bubble (used to skip self-intersection)
- `center_i::SVector{3, Float64}`: Center of the source bubble
- `n̂::SVector{3, Float64}`: Ray direction
- `nucleations::Vector`: Collection of all bubbles in the system (including periodic ghosts)
- `t_i::Float64`: Nucleation time of bubble i
- `t_end::Float64`: End of valid time window
- `v::Float64`: Bubble wall velocity

# Returns
Float64: Minimum collision time, or t_end - t_i if no collision occurs
"""
function find_collision_time(i::Int, center_i::SVector{3, Float64}, n̂::SVector{3, Float64},
                             nucleations::Vector, t_i::Float64, t_end::Float64, v::Float64)::Float64
    τ_min = Inf

    for (idx, nuc_j) in enumerate(nucleations)
        # Skip the exact source bubble
        idx == i && continue

        center_j = nuc_j[:site].coordinates
        t_j = nuc_j[:time]

        τ_coll = collision_time(center_i, center_j, n̂, t_i, t_j, v)

        # Update running minimum if a valid, earlier collision is found
        if τ_coll !== nothing && τ_coll < τ_min
            τ_min = τ_coll
        end
    end

    # Ray survives until the minimum of the collision time or the end of the simulation
    return min(τ_min, t_end - t_i)
end

function prepare_source_collision!(
    ws::SourceCollisionWorkspace,
    blockers::NucleationSoA,
    xi::Float64, yi::Float64, zi::Float64, ti::Float64,
    source_idx::Int,
    t_end::Float64,
    v::Float64,
)
    N = length(blockers.t)
    resize!(ws.dx, N)
    resize!(ws.dy, N)
    resize!(ws.dz, N)
    resize!(ws.dt, N)
    resize!(ws.numer, N)
    resize!(ws.tau_floor, N)
    empty!(ws.candidates)

    R_i_max = v * (t_end - ti)

    @inbounds for j in 1:N
        j == source_idx && continue

        dx = xi - blockers.x[j]
        dy = yi - blockers.y[j]
        dz = zi - blockers.z[j]
        dt = ti - blockers.t[j]

        R_j_max = v * (t_end - blockers.t[j])

        dist2 = dx*dx + dy*dy + dz*dz
        max_reach = R_i_max + R_j_max

        dist2 > max_reach^2 && continue # Filtering based on maximum reach: if the blocker is too far away to ever collide, skip.

        numer     = v*v*dt*dt - dist2

        ws.dx[j]        = dx
        ws.dy[j]        = dy
        ws.dz[j]        = dz
        ws.dt[j]        = dt
        ws.numer[j]     = numer
        ws.tau_floor[j] = max(0.0, -dt) # The ray cannot collide before the blocker nucleates, so τ must be at least t_j - t_i.

        push!(ws.candidates, j)
    end
end

function find_collision_time(n̂:: SVector{3,Float64},
                             ws::SourceCollisionWorkspace,
                             t_i::Float64, t_end::Float64, v::Float64)::Float64
    τ_min = t_end - t_i
    n1, n2, n3 = n̂

    @inbounds for j in ws.candidates
        ndotdx = n1*ws.dx[j] + n2*ws.dy[j] + n3*ws.dz[j]
        denom  = 2.0 * v * (ndotdx - v * ws.dt[j])

        denom >= -1.0e-12 && continue

        τ = ws.numer[j] / denom

        τ < ws.tau_floor[j] - 1.0e-12 && continue
        τ >= τ_min                      && continue

        τ_min = max(ws.tau_floor[j], τ)
    end

    return τ_min
end