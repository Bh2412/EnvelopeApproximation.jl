"""
    RayTracingStressEnergyTensor

Ray-tracing method for computing T_ij using deterministic spherical quadrature
and analytic time integration via the I₃(α; a, b) formula.

Key features:
- No dome-ring topology complexity
- No time quadrature (all integrals are closed-form)
- Pluggable spherical quadrature scheme interface
- Direct collision detection along rays

# References
The method discretizes the angular integral using spherical quadrature:
  ∫ dΩ f(n̂) ≈ ∑_a w_a f(n̂_a)

For each ray marker n̂_a, collision times are computed and the time integral
is evaluated analytically:
  A_ij^±(k) = (ΔV/3) ∑_a w_a n̂_{a,i} n̂_{a,j} e^{-ikz_i} e^{±ikt_i} v³ I₃(α_±; τ_start, τ_stop)

where I₃ is the closed-form integral of τ³ exp(iατ) from a to b.
"""
module RayTracingStressEnergyTensor

using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution: BubblesSnapShot, Bubble, Nucleation
using EnvelopeApproximation.Spaces: BoxSpace
using EnvelopeApproximation.BoundaryConditions: Periodic
using EnvelopeApproximation.EnvelopeAnalysis: append_periodic_bubbles!
import EnvelopeApproximation.StressEnergyTensorComponents: contribution_indices
using StaticArrays
using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

export SphericalQuadratureMarker, SphericalQuadratureScheme,
       RayTracingT_ij_CosineWeight,
       I3, collision_time, find_collision_time,
       UniformSphericalCapScheme, get_markers,
       ray_T_ij

# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Types & Interfaces
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SphericalQuadratureMarker

Represents a single ray direction with associated quadrature weight.

# Fields
- `n̂::SVector{3, Float64}`: Unit direction vector of the ray
- `weight::Float64`: Quadrature weight w_a
"""
struct SphericalQuadratureMarker
    n̂::SVector{3, Float64}
    weight::Float64
end

"""
    abstract type SphericalQuadratureScheme

Interface for spherical quadrature schemes. Implementations must provide:
  `get_markers(scheme::T)::Vector{SphericalQuadratureMarker}`
"""
abstract type SphericalQuadratureScheme end

"""
    RayTracingT_ij_CosineWeight

Strategy struct for ray-tracing T_ij computation with CosineWeight temporal decorrelation.

# Fields
- `quadrature::SphericalQuadratureScheme`: Spherical quadrature generator
- `markers::Vector{SphericalQuadratureMarker}`: Pre-computed ray markers (cached)
"""
struct RayTracingT_ij_CosineWeight
    quadrature::SphericalQuadratureScheme
    markers::Vector{SphericalQuadratureMarker}
end

function RayTracingT_ij_CosineWeight(quadrature::SphericalQuadratureScheme)
    return RayTracingT_ij_CosineWeight(quadrature, get_markers(quadrature))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Core: Analytic Integral I₃(α; a, b)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    I3(α::Float64, a::Float64, b::Float64) -> ComplexF64

Evaluate ∫_a^b τ³ exp(iατ) dτ analytically.
"""
function I3(α::Float64, a::Float64, b::Float64)::ComplexF64
    iszero(a) && return I3_zero_lower(α, b) # Fast path
    iszero(α) && return ComplexF64((b^4 - a^4) / 4.0)

    # Condition must check the maximum dimensionless phase, not just α.
    max_x = abs(α) * max(abs(a), abs(b))

    if max_x < 1.0e-4
        # ∫ τ³ (1 + iατ - α²τ²/2 - iα³τ³/6 + α⁴τ⁴/24) dτ
        a2, b2 = a*a, b*b
        a3, b3 = a2*a, b2*b
        a4, b4 = a2*a2, b2*b2
        a5, b5 = a4*a, b4*b
        a6, b6 = a5*a, b5*b
        a7, b7 = a6*a, b6*b
        a8, b8 = a7*a, b7*b
        
        return ComplexF64(
            (b4 - a4)/4.0 - α^2*(b6 - a6)/12.0 + α^4*(b8 - a8)/192.0,
            α*(b5 - a5)/5.0 - α^3*(b7 - a7)/42.0
        )
    end

    # General case: avoiding complex arithmetic
    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα
    invα4 = invα2 * invα2
    
    a2, b2 = a*a, b*b
    a3, b3 = a2*a, b2*b

    # A(τ) = 3τ²/α² - 6/α⁴
    # B(τ) = -τ³/α + 6τ/α³
    Aa = 3.0*a2*invα2 - 6.0*invα4
    Ba = -a3*invα + 6.0*a*invα3
    
    Ab = 3.0*b2*invα2 - 6.0*invα4
    Bb = -b3*invα + 6.0*b*invα3

    sa, ca = sincos(α*a)
    sb, cb = sincos(α*b)

    # F(τ) = (cos(ατ)A - sin(ατ)B) + i(sin(ατ)A + cos(ατ)B)
    return ComplexF64(
        (cb*Ab - sb*Bb) - (ca*Aa - sa*Ba),
        (sb*Ab + cb*Bb) - (sa*Aa + ca*Ba)
    )
end

@inline function I3_zero_lower(α::Float64, b::Float64)::ComplexF64
    # ∫₀ᵇ τ³ exp(i α τ) dτ

    b2 = b*b
    b3 = b2*b
    b4 = b2*b2

    x = abs(α * b)

    # Small phase: Taylor expansion in ατ.
    # ∫ τ³ ∑ₘ (iατ)^m/m! dτ
    if x < 1.0e-4
        b5 = b4*b
        b6 = b5*b
        b7 = b6*b
        b8 = b7*b

        return ComplexF64(
            b4/4 - α*α*b6/12 + α^4*b8/192,
            α*b5/5 - α^3*b7/42
        )
    end

    invα  = 1.0 / α
    invα2 = invα * invα
    invα3 = invα2 * invα
    invα4 = invα2 * invα2

    # Antiderivative:
    # e^{iαb}[b³/(iα) - 3b²/(iα)² + 6b/(iα)³ - 6/(iα)⁴] + 6/(iα)⁴
    #
    # Written as real arithmetic to avoid complex powers.
    A = 3.0*b2*invα2 - 6.0*invα4
    B = -b3*invα + 6.0*b*invα3

    s, c = sincos(α*b)

    return ComplexF64(
        c*A - s*B + 6.0*invα4,
        s*A + c*B
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Collision Time Computation
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# Structure-of-Arrays layout and per-source workspace for hot loop
# ═══════════════════════════════════════════════════════════════════════════════

struct NucleationSoA
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    t::Vector{Float64}
end

function nucleation_soa(nucleations)
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

function find_collision_time(n1::Float64, n2::Float64, n3::Float64,
                             ws::SourceCollisionWorkspace,
                             t_i::Float64, t_end::Float64, v::Float64)::Float64
    τ_min = t_end - t_i

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

# ═══════════════════════════════════════════════════════════════════════════════
# Quadrature Schemes
# ═══════════════════════════════════════════════════════════════════════════════

"""
    UniformSphericalCapScheme <: SphericalQuadratureScheme

Simple uniform spherical cap quadrature: N_θ rings × N_ϕ azimuthal points.

# Fields
- `N_theta::Int`: Number of polar angle points
- `N_phi::Int`: Number of azimuthal points per ring
"""
struct UniformSphericalCapScheme <: SphericalQuadratureScheme
    N_theta::Int
    N_phi::Int
end

function get_markers(scheme::UniformSphericalCapScheme)::Vector{SphericalQuadratureMarker}
    N_θ, N_ϕ = scheme.N_theta, scheme.N_phi
    
    markers = SphericalQuadratureMarker[]

    # Weight for unit sphere quadrature: each cell contributes Δμ × Δϕ 
    
    Δμ = 2.0 / N_θ
    Δϕ = 2.0 * π / N_ϕ
    w = Δμ * Δϕ 
    
    # Polar angles: cos(θ) ∈ [-1, 1] uniform in μ = cos(θ)
    for i in 1:N_θ

        μ = i * Δμ - 1.0 - Δμ / 2  
        s  = sqrt(clamp(1.0 - μ^2, 0.0, 1.0))  # sin(θ) for Cartesian conversion
        # Azimuthal angles
        for j in 1:N_ϕ
            ϕ = j * Δϕ - Δϕ / 2  # Centered in each azimuthal cell
            
            n̂ = SVector(s * cos(ϕ), s * sin(ϕ), μ)
            push!(markers, SphericalQuadratureMarker(n̂, w))
        end
    end
    
    return markers
end

# ═══════════════════════════════════════════════════════════════════════════════
# Ray-Tracing T_ij Core Computation
# ═══════════════════════════════════════════════════════════════════════════════

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

function nucleation_ray_T_ij_contribution!(ks::Vector{Float64},
                                           source_nucleation_idx::Int,
                                           source_nucleation::Nucleation,
                                           blockers_soa::NucleationSoA,
                                           ws::SourceCollisionWorkspace,
                                           markers::Vector{SphericalQuadratureMarker},
                                           t_end::Float64;
                                           ΔV::Float64=1.0, v::Float64=1., A_plus::Matrix{ComplexF64},
                                           A_minus::Matrix{ComplexF64})
    t_i      = source_nucleation[:time]
    center_i = source_nucleation[:site].coordinates
    xi, yi, zi = center_i[1], center_i[2], center_i[3]

    prepare_source_collision!(ws, blockers_soa, xi, yi, zi, t_i, source_nucleation_idx, t_end, v)

    amp_base = (ΔV / 3.0) * v^3

    phase_plus  = Vector{ComplexF64}(undef, length(ks))
    phase_minus = Vector{ComplexF64}(undef, length(ks))

    @inbounds for q in eachindex(ks)
        k = ks[q]
        phase_plus[q]  = amp_base * cis( k * ( t_i - zi))
        phase_minus[q] = amp_base * cis(-k * ( t_i + zi))
    end

    for marker in markers
        n̂ = marker.n̂
        w = marker.weight
        n1, n2, n3 = n̂[1], n̂[2], n̂[3]

        τ_stop = find_collision_time(n1, n2, n3, ws, t_i, t_end, v)

        τ_stop < 1.0e-12 && continue

        for (k_idx, k) in enumerate(ks)
            α_plus  = k * ( 1.0 - v * n3)
            α_minus = k * (-1.0 - v * n3)

            I3_plus  = I3(α_plus,  0.0, τ_stop)
            I3_minus = I3(α_minus, 0.0, τ_stop)

            amp_plus = w *  phase_plus[k_idx]  * I3_plus
            amp_minus = w * phase_minus[k_idx] * I3_minus

            A_plus[1, k_idx] += amp_plus  * n1 * n1
            A_plus[2, k_idx] += amp_plus  * n1 * n2
            A_plus[3, k_idx] += amp_plus  * n1 * n3
            A_plus[4, k_idx] += amp_plus  * n2 * n2
            A_plus[5, k_idx] += amp_plus  * n2 * n3
            A_plus[6, k_idx] += amp_plus  * n3 * n3

            A_minus[1, k_idx] += amp_minus * n1 * n1
            A_minus[2, k_idx] += amp_minus * n1 * n2
            A_minus[3, k_idx] += amp_minus * n1 * n3
            A_minus[4, k_idx] += amp_minus * n2 * n2
            A_minus[5, k_idx] += amp_minus * n2 * n3
            A_minus[6, k_idx] += amp_minus * n3 * n3
        end
    end
end

"""
Compute the time-integrated amplitudes

  A±_I(k) = (ΔV/3) ∑_i ∑_a w_a n̂_{a,I} e^{-ikzᵢ} e^{±iktᵢ} v³
            I₃(k(±1-vn̂_z); 0, τ_stop)

where I labels the six symmetric tensor components
xx=1, xy=2, xz=3, yy=4, yz=5, zz=6.

Returns two 6×Nk matrices, `A_plus` and `A_minus`, whose columns correspond
to individual k values.

For each k index q, the cosine-weighted tensor correlator is the 6×6 matrix

  D[:, :, q] =
      (1/(2V)) * (
          A_plus[:, q]  * A_plus[:, q]' +
          A_minus[:, q] * A_minus[:, q]'
      )

where `'` denotes complex conjugate transpose.
"""
function ray_T_ij(ks::AbstractVector{Float64}, snapshot::BubblesSnapShot,
                  space::BoxSpace, boundary_condition::Periodic,
                  strategy::RayTracingT_ij_CosineWeight;
                  ΔV::Float64=1.0, v::Float64=1., bubble_indices=:)::Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}
    ks_f = collect(Float64, ks)
    Nk = length(ks_f)

    empty_result = (zeros(ComplexF64, 6, Nk), zeros(ComplexF64, 6, Nk))
    isempty(snapshot.nucleations) && return empty_result

    A_plus  = zeros(ComplexF64, 6, Nk)
    A_minus = zeros(ComplexF64, 6, Nk)
    markers = strategy.markers

    t_end = snapshot.t
    full_nucleations = periodic_nucleations(snapshot.nucleations, v, 0.0, t_end, space)
    contributing_indices = contribution_indices(length(snapshot.nucleations), bubble_indices)

    blockers_soa = nucleation_soa(full_nucleations)
    ws = SourceCollisionWorkspace(length(full_nucleations))

    for idx in contributing_indices
        nuc = full_nucleations[idx]
        nucleation_ray_T_ij_contribution!(ks_f, idx, nuc, blockers_soa, ws, markers, t_end; ΔV, v, A_plus=A_plus, A_minus=A_minus)
    end
    
    return A_plus, A_minus
end

end # module RayTracingStressEnergyTensor
