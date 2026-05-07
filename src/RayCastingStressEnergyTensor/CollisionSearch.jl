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

function collision_time(center_i::SVector{3, Float64}, center_j::SVector{3, Float64}, n̂::SVector{3, Float64},
                        t_i::Float64, t_j::Float64, v::Float64)::Union{Float64, Nothing}
    Δx = center_i - center_j
    Δt = t_i - t_j

    denom = 2.0 * v * (dot(n̂, Δx) - v * Δt)
    denom > -1.0e-12 && return nothing

    numer = v^2 * Δt^2 - sum(abs2, Δx)
    τ = numer / denom

    τ_floor = max(0.0, -Δt)
    τ < τ_floor - 1.0e-12 && return nothing

    return max(τ_floor, τ)
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

function prepare_source_collision!(
    ws::SourceCollisionWorkspace,
    blockers::NucleationSoA,
    source_nucleation::Nucleation,
    source_idx::Int,
    t_end::Float64,
    v::Float64,
)
    center_i = source_nucleation[:site].coordinates
    return prepare_source_collision!(
        ws, blockers,
        center_i[1], center_i[2], center_i[3], source_nucleation[:time],
        source_idx, t_end, v,
    )
end

"""
Find the earliest collision time for a ray from bubble `i` along direction n̂.

Loops through all blockers to compute their collision time with the ray.
Returns the minimum one.

# Arguments
- `n̂::SVector{3, Float64}`: Ray direction
- `ws::SourceCollisionWorkspace`: Precomputed workspace containing blocker data, usually filtered
    for optimization.
- `t_i::Float64`: Nucleation time of the source bubble
- `t_end::Float64`: End of valid time window
- `v::Float64`: Bubble wall velocity

# Returns
Float64: Minimum collision time, or t_end - t_i if no collision occurs
"""

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
