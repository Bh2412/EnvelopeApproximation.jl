begin
    using EnvelopeApproximation.BubbleBasics: Bubble
    using EnvelopeApproximation.EnvelopeAnalysis: IntersectionDome, inanydome
    using StaticArrays
    using LinearAlgebra
    using HCubature
    using Random
    using Measurements
    using QuadGK
end

# Unit normal vector for given (μ, ϕ)
function n̂(μ::Real, ϕ::Real)::SVector{3,Float64}
    sin_theta = sqrt(1 - μ^2)
    n_y, n_x = sin_theta .* sincos(ϕ)
    n_z = μ
    return SVector(n_x, n_y, n_z)
end

function phases(x̂::SVector{3,Float64}, ks::AbstractArray, bubble::Bubble)::Vector{ComplexF64}
    translation = bubble.radius * x̂ + coordinates(bubble.center)
    return map(ks) do k
        cis(-k ⋅ translation)
    end
end

"""
    complex_to_real_covariance_optimized(mean_Z, Σ_H, Σ_P)

Efficiently converts complex statistics into real-valued formats for Measurements.jl.
Uses `reinterpret` to create the mean vector without allocation/copying.
"""
function complex_to_real_covariance_matrix(
    Σ_H::Matrix{ComplexF64},
    Σ_P::Matrix{ComplexF64}
)
    M = size(Σ_H, 1)
    S_plus = Σ_H + Σ_P
    S_minus = Σ_H - Σ_P

    cov_real = zeros(Float64, 2 * M, 2 * M)

    @inbounds for j in 1:M
        @inbounds for i in 1:j
            # Indices in the real matrix
            Xi, Yi = 2i - 1, 2i
            Xj, Yj = 2j - 1, 2j
            s_p = S_plus[i, j]
            s_m = S_minus[i, j]

            # 1. Cov(Xi, Xj) = 0.5 * Re(H + P)
            cov_real[Xi, Xj] = 0.5 * real(s_p)

            # 2. Cov(Yi, Yj) = 0.5 * Re(H - P)
            cov_real[Yi, Yj] = 0.5 * real(s_m)

            # 3. Cov(Xi, Yj) = 0.5 * Im(P - H)
            cov_real[Xi, Yj] = -0.5 * imag(s_m)

            # 4. Cov(Yi, Xj) = 0.5 * Im(P + H)
            # For the i == j case, this write to the lower triangle. When calling "Symmetric" later on we "run-over" this value.
            cov_real[Yi, Xj] = 0.5 * imag(S_plus[i, j])
        end
    end
    return Symmetric(cov_real, :U)
end

function correlated_complex_measurements(
    mean_Z::Vector{ComplexF64},
    Σ_H::Matrix{ComplexF64},
    Σ_P::Matrix{ComplexF64}
)
    M = length(mean_Z)

    # Convert mean to real-valued without allocation
    mean_real = reinterpret(Float64, mean_Z)

    # Convert covariance matrices
    cov_real = complex_to_real_covariance_matrix(Σ_H, Σ_P)

    # Create Measurements
    measurements_real = Measurements.correlated_values(mean_real, cov_real)
    # Reinterpret back to complex Measurements
    measurements_complex = similar(mean_Z, Complex{Measurement{Float64}})

    @inbounds for i in 1:M
        re = measurements_real[2i-1]
        im = measurements_real[2i]
        measurements_complex[i] = complex(re, im)
    end

    return measurements_complex
end

"""
    ensemble_monte_carlo_sum(f, K, bubbles, domes_dict, ks; N_samples, rng)

Estimates the sum of integrals over all bubbles: S = ∑_i ∫ dΩ_i f(...)
by sampling bubbles proportional to R^3 -> This assumes the contrtibution of each bubble is proportional to R^3.
"""
function bubble_envelope_monte_carlo_fourier_transform(
    f,
    K::Int,
    bubbles::Vector{Bubble},
    domes_dict::Dict{Int,Vector{IntersectionDome}},
    ks::AbstractArray;
    prefactor::Float64=1.,
    N_samples::Int=1_000_000,
    rng::AbstractRNG=Random.default_rng()
)
    ks_size = size(ks)
    N_k = length(ks)

    # 1. Setup Bubble Selection Probabilities (Proportional to R^3)
    weights = map(b -> b.radius^3, bubbles)
    prefactor = begin
        total_weight = sum(weights)
        4π * total_weight * prefactor
    end # This prefactor transforms the result from an average to an integral over the domain

    # Create a sampler for efficient selection
    # (Aliasing or simple categorical sampling)
    bubble_indices = 1:length(bubbles)
    bubble_sampler = Weights(weights) # StatsBase handles the normalization

    # Accumulators
    N_data = K * N_k
    # From the following we can construct the covariance matrix of the results
    val = zeros(ComplexF64, N_data) # Temporary storage for each sample's result
    S_1 = zeros(ComplexF64, N_data) # Nominal complex sum
    S_H = zeros(ComplexF64, N_data, N_data) # Z * Z†
    S_P = zeros(ComplexF64, N_data, N_data) # Z * ZT

    # 2. Monte Carlo Loop
    for _ in 1:N_samples
        # A. Sample a bubble index 'i' with probability P_i
        idx = sample(rng, bubble_indices, bubble_sampler)
        bubble = bubbles[idx]
        domes = domes_dict[idx]

        # B. Sample a point (μ, ϕ) uniformly on the sphere
        # Volume of sample space = 4π
        μ = 2.0 * rand(rng) - 1.0
        ϕ = 2π * rand(rng)
        x̂ = n̂(μ, ϕ)

        # C. Check geometric overlap
        if inanydome(x̂, bubble, domes)
            continue # Contribution is 0
        end

        # D. Evaluate Integrand
        f_val = f(μ, ϕ) * prefactor
        p_val = phases(x̂, ks, bubble)
        ptr = 1
        @inbounds for k_i in eachindex(ks)
            p = p_val[k_i]
            for c_i in 1:K
                val[ptr] = f_val[c_i] * p
                ptr += 1
            end
        end

        S_1 .+= val
        BLAS.herk!('U', 'N', 1.0, val, 1.0, S_H) # S_H += val * val†
        BLAS.syr!('U', 1.0, val, S_P) # S_P += val * valT
    end

    # 3. Final Statistics
    mean_Z = S_1 ./ N_samples
    Σ_H = (S_H - (1 / N_samples) * (S_1 * S_1')) ./ (N_samples * (N_samples - 1)) # Covariance matrix
    Σ_P = (S_P - (1 / N_samples) * (S_1 * transpose(S_1))) ./ (N_samples * (N_samples - 1)) # Pseudo Covariance matrix

    complex_measurement = correlated_complex_measurements(mean_Z, Σ_H, Σ_P)
    return reshape(complex_measurement, K, ks_size...)
end

function x̂ᵢx̂ⱼ(μ::Real, ϕ::Real)::SVector{6,Float64}
    # Pre-calculate powers and roots
    μ² = μ^2
    s² = clamp(1 - μ², -1., 1.)
    s = sqrt(s²)
    return SVector{6,Float64}(s^2 * cos(ϕ)^2,  # xx
        s^2 * sin(ϕ) * cos(ϕ), # xy
        s * μ * cos(ϕ),        # xz
        s^2 * sin(ϕ)^2,  # yy
        s * μ * sin(ϕ),        # yz
        μ²)              # zz

end

function mc_∂ᵢϕ∂ⱼϕ(ks::AbstractArray, bubbles::AbstractVector{Bubble}, ball_space::BallSpace;
    N_samples::Int=1_000_000, rng::AbstractRNG=Random.default_rng(), ΔV::Float64=1.0)
    domes = intersection_domes(bubbles, ball_space)
    return bubble_envelope_monte_carlo_fourier_transform(
        x̂ᵢx̂ⱼ,
        6,
        bubbles,
        domes,
        ks;
        prefactor=ΔV / 3.,
        N_samples=N_samples,
        rng=rng
    )
end

"""
    compute_Lambda_matrix(k::AbstractVector)

Computes the 6x6 matrix representation of the Lambda projection tensor Λ_ijlm
for a generic propagation direction k.

The matrix operates on the flattened symmetric tensor vector:
v = [xx, xy, xz, yy, yz, zz]
"""
function compute_Lambda_matrix(k::AbstractVector):: SMatrix{6,6,Float64}
    # 1. Normalize k to get k̂
    k̂ = normalize(k)

    # 2. Define the Projection Tensor P_ab = δ_ab - k̂_a * k̂_b
    function P(a, b)
        δ = (a == b) ? 1.0 : 0.0
        return δ - k̂[a] * k̂[b]
    end

    # 3. Define the Raw Lambda Tensor (Not generally symmetric in l,m)
    # Λ_ijlm = P_il * P_jm - 0.5 * P_ij * P_lm
    function Λ_raw(i, j, l, m)
        return P(i, l) * P(j, m) - 0.5 * P(i, j) * P(l, m)
    end

    # 4. Indices for the 6-component vector
    # Order: xx, xy, xz, yy, yz, zz
    indices = [
        (1, 1), # 1: xx
        (1, 2), # 2: xy
        (1, 3), # 3: xz
        (2, 2), # 4: yy
        (2, 3), # 5: yz
        (3, 3)  # 6: zz
    ]

    # 5. Construct Matrix
    mat = Matrix{Float64}(undef, 6, 6)

    for (row, (i, j)) in enumerate(indices)
        for (col, (l, m)) in enumerate(indices)

            if l == m
                # Diagonal input (e.g., T_xx): Just one term in the sum
                mat[row, col] = Λ_raw(i, j, l, m)
            else
                # Off-diagonal input (e.g., T_xy):
                # The summation over l,m hits both (1,2) and (2,1).
                # Coefficient is Λ_ij12 + Λ_ij21
                mat[row, col] = Λ_raw(i, j, l, m) + Λ_raw(i, j, m, l)
            end
        end
    end

    return SMatrix{6,6,Float64}(mat)
end

# Weight matrix for tensor contraction in 6-component basis
# Indices: 1:xx, 2:xy, 3:xz, 4:yy, 5:yz, 6:zz
const W_tensor_contraction = Diagonal(SVector{6,Float64}(1., 2., 2., 1., 2., 1.))

"""
    Π_MC(t1, t2, ks, snapshot, ball_space; ...)

Computes the anisotropic stress power spectrum Π(k) using Monte Carlo integration.
Unlike the semi-analytic version, this accepts vector `ks` and handles the general 
tensor projection for each direction.

# Arguments
- `ks`: AbstractVector of 3D k-vectors (e.g. `[Vec3(0,0,1), ...]`).
"""
function Π_MC(
    t1::Float64,
    t2::Float64,
    ks::AbstractVector{<:SVector{3,Float64}},
    snapshot::BubblesSnapShot,
    ball_space::BallSpace;
    N_samples::Int=100_000,
    rng::AbstractRNG=Random.default_rng(),
    ΔV::Float64=1.0
)
    # 1. Compute T_ij via MC
    if t1 ≈ t2
        T1 = mc_∂ᵢϕ∂ⱼϕ(ks, bubbles_snapshot_t1, ball_space; N_samples=N_samples, rng=rng, ΔV=ΔV) # (Pseudo-code for bubbles input)
        # Note: You need to extract bubbles from snapshot first as in your previous snippet:
        # bubbles = current_bubbles(snapshot, t1)
        # T1 = mc_∂ᵢϕ∂ⱼϕ(ks, bubbles, ...)
        T2 = T1
    else
        bubbles1 = current_bubbles(snapshot, t1)
        bubbles2 = current_bubbles(snapshot, t2)
        T1 = mc_∂ᵢϕ∂ⱼϕ(ks, bubbles1, ball_space; N_samples=N_samples, rng=rng, ΔV=ΔV)
        T2 = mc_∂ᵢϕ∂ⱼϕ(ks, bubbles2, ball_space; N_samples=N_samples, rng=rng, ΔV=ΔV)
    end

    # 2. Contract
    V = volume(ball_space)

    # Pre-allocate output with exact shape of ks
    Π_vals = similar(ks, Complex{Measurement{Float64}})

    @inbounds for i in eachindex(ks)
        k_vec = ks[i]

        # Extract columns as SVector for fast linear algebra
        v1 = SVector{6}(view(T1, :, i))
        v2 = SVector{6}(view(T2, :, i))

        # Compute Projection Matrix M(k)
        Λ = compute_Lambda_matrix(k_vec)

        # v2' * W * Λ * v1
        Λ_projected = Λ * v1
        weighted_Λ_projected = W_tensor_contraction * Λ_projected
        Π_vals[i] = dot(v2, weighted_Λ_projected) / V
    end

    return Π_vals
end
