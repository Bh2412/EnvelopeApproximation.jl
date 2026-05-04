module StressEnergyTensorContractions

using LinearAlgebra
using StaticArrays

export symmetric_tensor_pairs,
       contraction_weights,
       Λ_TT, Λ_T, Λ_pp, Λ_σσ, Λ_pσ,
       Lambda_TT, Lambda_T, Lambda_pp, Lambda_σσ, Lambda_pσ,
       contraction_matrix,
       contract,
       all_contractions

const symmetric_tensor_pairs = SVector(
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
)

const contraction_weights = Diagonal(SVector{6,Float64}(1.0, 2.0, 2.0, 1.0, 2.0, 1.0))

@inline δ(i::Integer, j::Integer) = i == j ? 1.0 : 0.0

function unit_direction(k::AbstractVector{<:Real})::SVector{3,Float64}
    @assert length(k) == 3 "Wave-vector direction must have three components"
    kvec = SVector{3,Float64}(k)
    k_norm = norm(kvec)
    @assert k_norm > 0 "Wave-vector direction must be non-zero"
    return kvec / k_norm
end

function projector_matrix(Λ_raw)::SMatrix{6,6,Float64}
    mat = Matrix{Float64}(undef, 6, 6)
    for (row, (i, j)) in enumerate(symmetric_tensor_pairs)
        for (col, (m, n)) in enumerate(symmetric_tensor_pairs)
            mat[row, col] = m == n ? Λ_raw(i, j, m, n) : Λ_raw(i, j, m, n) + Λ_raw(i, j, n, m)
        end
    end
    return SMatrix{6,6,Float64}(mat)
end

"""
    Λ_TT(k)

Transverse-traceless tensor contractor for gravitational-wave amplitudes,
stored in the six-component symmetric tensor basis
`(11, 12, 13, 22, 23, 33)`.
"""
function Λ_TT(k::AbstractVector{<:Real})::SMatrix{6,6,Float64}
    k̂ = unit_direction(k)
    P(a, b) = δ(a, b) - k̂[a] * k̂[b]
    return projector_matrix((i, j, m, n) -> P(i, m) * P(j, n) - 0.5 * P(i, j) * P(m, n))
end

"""
    Λ_T(k)

Vector contractor `P_im k_j k_n`.
"""
function Λ_T(k::AbstractVector{<:Real})::SMatrix{6,6,Float64}
    k̂ = unit_direction(k)
    absksquared = norm(k) ^ 2
    P(a, b) = δ(a, b) - k̂[a] * k̂[b]
    return projector_matrix((i, j, m, n) -> P(i, m) * k̂[j] * k̂[n]) / absksquared
end

"""
    Λ_pp([k])

Pressure-pressure contractor `(1/9) δ_ij δ_mn`.
The optional argument is accepted for API symmetry with direction-dependent contractors.
"""
Λ_pp()::SMatrix{6,6,Float64} = projector_matrix((i, j, m, n) -> (1.0 / 9.0) * δ(i, j) * δ(m, n))
Λ_pp(::AbstractVector{<:Real})::SMatrix{6,6,Float64} = Λ_pp()

"""
    Λ_σσ(k)

Anisotropic stress auto-correlation contractor
`(9/4) (δ_ij/3 - k_i k_j) (δ_mn/3 - k_m k_n)`.
"""
function Λ_σσ(k::AbstractVector{<:Real})::SMatrix{6,6,Float64}
    k̂ = unit_direction(k)
    L(a, b) = δ(a, b) / 3.0 - k̂[a] * k̂[b]
    return projector_matrix((i, j, m, n) -> (9.0 / 4.0) * L(i, j) * L(m, n))
end

"""
    Λ_pσ(k)

Mixed pressure-anisotropic-stress contractor
`(1/2) (δ_ij/3 - k_i k_j) δ_mn`.
"""
function Λ_pσ(k::AbstractVector{<:Real})::SMatrix{6,6,Float64}
    k̂ = unit_direction(k)
    L(a, b) = δ(a, b) / 3.0 - k̂[a] * k̂[b]
    return projector_matrix((i, j, m, n) -> 0.5 * L(i, j) * δ(m, n))
end

const Lambda_TT = Λ_TT
const Lambda_T = Λ_T
const Lambda_pp = Λ_pp
const Lambda_σσ = Λ_σσ
const Lambda_pσ = Λ_pσ

"""
    contraction_matrix(Λ)

Return the weighted matrix used to contract a six-by-six stored
`<T_ij T_mn>` matrix with `tr(contraction_matrix(Λ) * T)`.
"""
contraction_matrix(Λ::AbstractMatrix) = Matrix(contraction_weights) * Matrix(Λ)

"""
    contract(T, Λ)
    contract(T, contractor, k)

Contract a stored six-by-six two-point stress-energy tensor matrix using one
of the `Λ_*` projector matrices or contractor functions.
"""
contract(T::AbstractMatrix, Λ::AbstractMatrix) = tr(contraction_matrix(Λ) * T)
contract(T::AbstractMatrix, contractor::Function, k::AbstractVector{<:Real}) = contract(T, contractor(k))

all_contractions(T::AbstractMatrix, k::AbstractVector{<:Real}) = (
    TT = contract(T, Λ_TT, k),
    T = contract(T, Λ_T, k),
    pp = contract(T, Λ_pp, k),
    σσ = contract(T, Λ_σσ, k),
    pσ = contract(T, Λ_pσ, k),
)

end
