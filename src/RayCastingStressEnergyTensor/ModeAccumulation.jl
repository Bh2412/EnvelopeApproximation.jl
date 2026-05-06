import Base.resize!

mutable struct SourceModeWorkspace
    phase_plus::Vector{ComplexF64}
    phase_minus::Vector{ComplexF64}
end

SourceModeWorkspace(Nk::Int) =
    SourceModeWorkspace(Vector{ComplexF64}(undef, Nk),
                        Vector{ComplexF64}(undef, Nk))

function resize!(ws::SourceModeWorkspace, Nk::Int)
    resize!(ws.phase_plus, Nk)
    resize!(ws.phase_minus, Nk)
    return ws
end

function prepare_source_phases!(
    ws::SourceModeWorkspace,
    ks,
    t_i::Float64,
    zi::Float64,
    amp_base::Float64,
)
    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])
        ws.phase_plus[q]  = amp_base * cis( k * ( t_i - zi))
        ws.phase_minus[q] = amp_base * cis(-k * ( t_i + zi))
    end
    return nothing
end

@inline function accumulate_tensor_components!(
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
    q::Int,
    amp_plus::ComplexF64,
    amp_minus::ComplexF64,
    n1::Float64,
    n2::Float64,
    n3::Float64,
)
    n11 = n1*n1
    n12 = n1*n2
    n13 = n1*n3
    n22 = n2*n2
    n23 = n2*n3
    n33 = n3*n3

    @inbounds begin
        A_plus[1,q] += amp_plus  * n11
        A_plus[2,q] += amp_plus  * n12
        A_plus[3,q] += amp_plus  * n13
        A_plus[4,q] += amp_plus  * n22
        A_plus[5,q] += amp_plus  * n23
        A_plus[6,q] += amp_plus  * n33

        A_minus[1,q] += amp_minus * n11
        A_minus[2,q] += amp_minus * n12
        A_minus[3,q] += amp_minus * n13
        A_minus[4,q] += amp_minus * n22
        A_minus[5,q] += amp_minus * n23
        A_minus[6,q] += amp_minus * n33
    end

    return nothing
end

function accumulate_marker_modes!(
    A_plus::Matrix{ComplexF64},
    A_minus::Matrix{ComplexF64},
    ks::AbstractVector{<:Real},
    τ_stop::Float64,
    n1::Float64,
    n2::Float64,
    n3::Float64,
    w::Float64,
    phase_plus::AbstractVector{ComplexF64},
    phase_minus::AbstractVector{ComplexF64},
    v::Float64,
)
    c_plus  =  1.0 - v*n3
    c_minus = -1.0 - v*n3

    @inbounds for q in eachindex(ks)
        k = Float64(ks[q])

        I3_plus  = I3(k*c_plus,  0.0, τ_stop)
        I3_minus = I3(k*c_minus, 0.0, τ_stop)

        amp_plus  = w * phase_plus[q]  * I3_plus
        amp_minus = w * phase_minus[q] * I3_minus

        accumulate_tensor_components!(
            A_plus, A_minus, q,
            amp_plus, amp_minus,
            n1, n2, n3,
        )
    end

    return nothing
end
