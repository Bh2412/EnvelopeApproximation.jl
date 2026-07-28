module RayCastingEnvelopeIntegration

using StaticArrays
using LinearAlgebra

import EnvelopeApproximation:
    Kernel, allocate_accumulant, prepare_kernel!, accumulate_ray!, finalize_accumulant!
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Periodic
using EnvelopeApproximation.EnvelopeAnalysis: append_periodic_bubbles!

include("CollisionSearch.jl")
include("SphericalQuadratureScheme.jl")

export LightConeSource,
       LightConeSurfaceContext,
       lightcone_sources,
       build_lightcone_context,
       integrate_lightcone_surfaces!,
       integrate_lightcone_surfaces,
       ray_stop_time,
       SphericalQuadratureScheme,
       SphericalQuadratureMarker,
       UniformSphericalCapScheme,
       get_markers

# =============================================================================
# 1. Source representation
# =============================================================================

struct LightConeSource
    id::Int
    time::Float64
    center::SVector{3,Float64}
end

function lightcone_sources(snapshot::BubblesSnapShot; bubble_indices = :)
    inds = bubble_indices isa Colon ? eachindex(snapshot.nucleations) : bubble_indices
    return map(inds) do i
        nuc = snapshot.nucleations[i]
        LightConeSource(
            Int(i),
            Float64(nuc.time),
            SVector{3,Float64}(coordinates(nuc.site)),
        )
    end
end

# =============================================================================
# 2. Snapshot-derived light-cone context
# =============================================================================

struct LightConeSurfaceContext{Snapshot,Space,BC,Blockers,Workspace}
    snapshot::Snapshot
    space::Space
    boundary_condition::BC
    blockers::Blockers
    workspace::Workspace
    t_end::Float64
    v::Float64
end

function build_lightcone_context(
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    boundary_condition;
    v::Float64 = 1.0,
)
    blockers = build_lightcone_blockers(snapshot, space, boundary_condition; v=v)
    workspace = allocate_lightcone_workspace(blockers)
    return LightConeSurfaceContext(
        snapshot,
        space,
        boundary_condition,
        blockers,
        workspace,
        Float64(snapshot.t),
        v,
    )
end

# =============================================================================
# 3. Collision/search interface
# =============================================================================

function prepare_source!(
    context::LightConeSurfaceContext,
    source::LightConeSource,
)
    prepare_source_collision!(
        context.workspace,
        context.blockers,
        source,
        context.t_end,
        context.v,
    )
    return nothing
end

function ray_stop_time(
    context::LightConeSurfaceContext,
    source::LightConeSource,
    n̂::SVector{3,Float64},
)::Float64
    return first_collision_time(
        context.workspace,
        source,
        n̂,
        context.t_end,
        context.v,
    )
end

# =============================================================================
# 4. Generic light-cone surface integrator
# =============================================================================

function integrate_lightcone_surfaces!(
    accs::Tuple,
    kernels::Tuple{Vararg{Kernel}},
    sources,
    context::LightConeSurfaceContext,
    markers;
    τ_min::Float64 = 0.0,
)
    @assert length(accs) == length(kernels)

    for source in sources
        prepare_source!(context, source)
        foreach(kernel -> prepare_kernel!(kernel, source), kernels)

        for marker in markers
            n̂ = marker.n̂
            wΩ = marker.weight

            τ_stop = ray_stop_time(context, source, n̂)
            τ_stop > τ_min || continue

            @inbounds for a in eachindex(kernels)
                accumulate_ray!(accs[a], kernels[a], source, n̂, wΩ, τ_stop)
            end
        end
    end

    @inbounds for a in eachindex(kernels)
        finalize_accumulant!(accs[a], kernels[a])
    end

    return accs
end

function integrate_lightcone_surfaces(
    kernels::Tuple{Vararg{Kernel}},
    sources,
    context::LightConeSurfaceContext,
    markers;
    kwargs...,
)
    accs = map(allocate_accumulant, kernels)
    return integrate_lightcone_surfaces!(accs, kernels, sources, context, markers; kwargs...)
end

# =============================================================================
# 5. Implementations of the geometric interface
# =============================================================================

function build_lightcone_blockers(
    snapshot::BubblesSnapShot,
    space::BoxSpace,
    ::Periodic;
    v::Float64 = 1.0,
)
    isempty(snapshot.nucleations) && return NucleationSoA(Nucleation[])

    t_end = Float64(snapshot.t)
    original_times = map(nuc -> nuc[:time], snapshot.nucleations)
    original_centers = map(nuc -> nuc[:site].coordinates, snapshot.nucleations)

    bubbles = Bubble[]
    sizehint!(bubbles, length(original_centers))

    D_ray_max = v * (t_end - original_times[1])
    for i in eachindex(original_centers)
        R_b_max = v * (t_end - original_times[i])
        R_pad = D_ray_max + R_b_max
        push!(bubbles, Bubble(Point3(original_centers[i]), R_pad))
    end

    origin_map = Int[]
    padded_bubbles = append_periodic_bubbles!(bubbles, space, origin_map)

    padded_times = original_times[origin_map]
    nucleations =  map((t, b) -> (time=t, site=b.center), padded_times, padded_bubbles)
    return NucleationSoA(nucleations)
end

function allocate_lightcone_workspace(blockers::NucleationSoA)
    return SourceCollisionWorkspace(length(blockers.t))
end

function prepare_source_collision!(
    ws::SourceCollisionWorkspace,
    blockers::NucleationSoA,
    source::LightConeSource,
    t_end::Float64,
    v::Float64,
)
    return prepare_source_collision!(
        ws, blockers,
        source.center[1], source.center[2], source.center[3], source.time,
        source.id, t_end, v,
    )
end

function first_collision_time(
    workspace::SourceCollisionWorkspace,
    source::LightConeSource,
    n̂::SVector{3,Float64},
    t_end::Float64,
    v::Float64,
)::Float64
    return find_collision_time(n̂, workspace, source.time, t_end, v)
end

end # module RayCastingEnvelopeIntegration
