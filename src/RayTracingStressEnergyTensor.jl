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
    I3(α, a, b) -> ComplexF64

Evaluate ∫_a^b τ³ exp(iατ) dτ analytically.

For α = 0:  (b⁴ − a⁴) / 4

For |α| < 1e-10: Taylor series to avoid cancellation in the exact formula.

For α ≠ 0: antiderivative via integration by parts,
  F(τ) = exp(iατ) [τ³/(iα) − 3τ²/(iα)² + 6τ/(iα)³ − 6/(iα)⁴],
  result = F(b) − F(a).
"""
function I3(α::Float64, a::Float64, b::Float64)::ComplexF64
    # Fast path: α exactly zero
    iszero(α) && return ComplexF64((b^4 - a^4) / 4.0)

    # Small |α| regime: Taylor series ∫τ³(1 + iατ - α²τ²/2 + ...)dτ avoids cancellation
    if abs(α) < 1.0e-10
        int_0 = (b^4 - a^4) / 4.0
        int_1 = im * α * (b^5 - a^5) / 5.0
        int_2 = -α^2 * (b^6 - a^6) / 12.0
        return int_0 + int_1 + int_2
    end

    # General case: antiderivative via integration by parts.
    # ∫ τ³ exp(iατ) dτ = exp(iατ) [τ³/(iα) − 3τ²/(iα)² + 6τ/(iα)³ − 6/(iα)⁴]
    iα_inv = 1.0 / (im * α)

    result_b = cis(α * b) * (b^3 * iα_inv - 3*b^2 * iα_inv^2 + 6*b * iα_inv^3 - 6*iα_inv^4)
    result_a = cis(α * a) * (a^3 * iα_inv - 3*a^2 * iα_inv^2 + 6*a * iα_inv^3 - 6*iα_inv^4)

    return result_b - result_a
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
                                           nucleations::Vector{Nucleation},
                                           markers::Vector{SphericalQuadratureMarker},
                                           t_end::Float64;
                                           ΔV::Float64=1.0, v::Float64=1., A_plus::Matrix{ComplexF64},
                                           A_minus::Matrix{ComplexF64})
    t_i    = source_nucleation[:time]
    center_i = source_nucleation[:site].coordinates
    z_i    = center_i[3]
    for marker in markers
        n̂ = marker.n̂
        w = marker.weight

        τ_stop = find_collision_time(source_nucleation_idx, center_i, n̂, nucleations, t_i, t_end, v)

        # τ_start = 0 (bubble i just nucleated); skip if no time window
        if τ_stop < 1.0e-12
            continue
        end

        for (k_idx, k) in enumerate(ks)
            α_plus  = k * ( 1.0 - v * n̂[3])
            α_minus = k * (-1.0 - v * n̂[3])

            I3_plus  = I3(α_plus,  0.0, τ_stop)
            I3_minus = I3(α_minus, 0.0, τ_stop)

            phase_z = cis(-k * z_i)
            amp = (ΔV / 3.0) * v^3 * w * phase_z

            amp_plus  = amp * cis(k * t_i) * I3_plus
            amp_minus = amp * cis(-k * t_i) * I3_minus

            n1, n2, n3 = n̂[1], n̂[2], n̂[3]
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

    for idx in contributing_indices
        nuc = full_nucleations[idx]
        nucleation_ray_T_ij_contribution!(ks_f, idx, nuc, full_nucleations, markers, t_end; ΔV, v, A_plus=A_plus, A_minus=A_minus)
    end
    
    return A_plus, A_minus
end

end # module RayTracingStressEnergyTensor
