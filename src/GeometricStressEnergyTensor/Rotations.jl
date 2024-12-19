const NullVec:: Vec3 = Vec3(zeros(3))

const ẑ:: Vec3 = Vec3(0., 0., 1.)

∥(u:: Vec3, v:: Vec3):: Bool = u×v ≈ NullVec

function ∠(k:: Vec3):: Vec3
    (k ∥ ẑ) && return Vec3(0., 0., 0.)
    k_ = norm(k)
    θ = acos(k[3] / k_)
    return ((k / (k_ * sin(θ))) * θ) × ẑ
end

align_ẑ(k:: Vec3):: SMatrix{3, 3, Float64} = SMatrix{3, 3, Float64}(RotationVec(∠(k)...))

export align_ẑ

# The mapping between a 3 x 3 symmetric tensor's double indices and 
#  a vector of length 6
const SymmetricTensorMapping:: Dict{Int, Tuple{Int, Int}} = Dict(1 => (1, 1), 2 => (1, 2), 3 => (1, 3), 4 => (2, 2), 5 => (2, 3), 6 => (3, 3))

# The mapping looks like This:
#=
   1       2     3
 #undef    4     5
 #undef  #undef  6
=#


#=
This matrix applies the transformation law:
x̂_ix̂_j = R_li * Rmj * x̂′_l x̂′_m
=#

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
