"""
   NullVec::Vec3

A constant representing the zero vector (0, 0, 0).
"""
const NullVec:: Vec3 = Vec3(zeros(3))

"""
   ẑ::Vec3

A constant representing the unit vector in the z-direction (0, 0, 1).
"""
const ẑ:: Vec3 = Vec3(0., 0., 1.)

"""
   ∥(u::Vec3, v::Vec3)::Bool

Tests if two vectors are parallel.

# Arguments
- `u::Vec3`: The first vector
- `v::Vec3`: The second vector

# Returns
`true` if the cross product of the vectors is approximately the zero vector,
indicating they are parallel, `false` otherwise.
"""
∥(u:: Vec3, v:: Vec3):: Bool = u×v ≈ NullVec

"""
    ∠(k::Vec3)::Vec3

Computes the rotation vector needed to align a vector with the z-axis.

# Arguments
- `k::Vec3`: The vector to align with the z-axis

# Returns
A rotation vector (axis-angle representation) that, when applied,
will align `k` with the z-axis.

# Notes
- If `k` is already parallel to the z-axis, returns the zero vector (no rotation needed).
- The returned vector's magnitude is the rotation angle, and its direction is the rotation axis.
- This function produces an axis-angle representation (also known as a Rodrigues vector).

# See Also
- [`RotationVec`](@ref Rotations.RotationVec): Constructor from Rotations.jl that converts an axis-angle representation to a rotation matrix.
- [`align_ẑ`](@ref): Function that directly returns the rotation matrix for z-axis alignment.
"""
function ∠(k:: Vec3):: Vec3
    (k ∥ ẑ) && return Vec3(0., 0., 0.)
    k_ = norm(k)
    θ = acos(k[3] / k_)
    return ((k / (k_ * sin(θ))) * θ) × ẑ
end

"""
    align_ẑ(k::Vec3)::SMatrix{3, 3, Float64}

Creates a rotation matrix that aligns a vector with the z-axis.

# Arguments
- `k::Vec3`: The vector to align with the z-axis

# Returns
A 3×3 rotation matrix that, when applied to `k`, will align it with the z-axis.

# Notes
Uses the axis-angle representation (Rodrigues vector) computed by `∠` to create
the rotation matrix via `Rotations.RotationVec`.

# See Also
- [`∠`](@ref): Function that computes the rotation vector for z-axis alignment.
- [`RotationVec`](@ref Rotations.RotationVec): Constructor from Rotations.jl that converts an axis-angle representation to a rotation matrix.
"""
align_ẑ(k:: Vec3):: SMatrix{3, 3, Float64} = SMatrix{3, 3, Float64}(RotationVec(∠(k)...))

export align_ẑ

"""
   SymmetricTensorMapping::Dict{Int, Tuple{Int, Int}}

A mapping between the flattened indices of a symmetric tensor and its matrix indices.

The mapping converts between the 6-component vector representation of a symmetric 
3×3 tensor and its matrix indices. The mapping is:

1. (1,1) - xx component
2. (1,2) - xy component
3. (1,3) - xz component
4. (2,2) - yy component
5. (2,3) - yz component
6. (3,3) - zz component

This follows the convention:
  1       2     3
#undef    4     5
#undef  #undef  6
"""
const SymmetricTensorMapping:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2 => (1, 2), 3 => (1, 3), 4 => (2, 2), 5 => (2, 3), 6 => (3, 3))

"""
    symmetric_tensor_inverse_rotation(rotation::SMatrix{3, 3, Float64})::SMatrix{6, 6, Float64}

Computes the transformation matrix for a symmetric tensor under coordinate rotation.

Given a rotation matrix `R`, this function computes the 6×6 matrix that transforms
the components of a symmetric tensor from one coordinate system to another.

This implements the tensor transformation law:
    x̂ᵢx̂ⱼ = Rₗᵢ Rₘⱼ x̂′ₗx̂′ₘ

where x̂ are the basis vectors in the original coordinate system and x̂′ are the 
basis vectors in the rotated coordinate system.

# Arguments
- `rotation::SMatrix{3, 3, Float64}`: The 3×3 rotation matrix

# Returns
A 6×6 matrix that transforms the vector representation of a symmetric tensor
under the given rotation.

# See Also
- [`SymmetricTensorMapping`](@ref): The mapping between flattened indices and matrix indices.
"""
function symmetric_tensor_inverse_rotation(rotation:: SMatrix{3, 3, Float64}):: SMatrix{6, 6, Float64}
    drot = MMatrix{6, 6, Float64}(undef)
    @inbounds for n ∈ 1:6, k ∈ 1:6
        i, j = SymmetricTensorMapping[k]
        l, m = SymmetricTensorMapping[n]
        if l == m
            drot[k, n] = rotation[l, i] * rotation[m, j] 
        else
            drot[k, n] = (rotation[l, i] * rotation[m, j]) + (rotation[m, i] * rotation[l, j]) 
        end
    end
    return drot
end

export symmetric_tensor_inverse_rotation
