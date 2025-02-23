module FilonQuadrature

using EnvelopeApproximation.FractionalFFT

    # Following "An Algorithm for Filon Quadrature" - STEPHEN M . CHASE AND LLOYD D. FOSDICK

const ε_θ:: Float64 = 1. / 6

function αβγ(θ:: Float64):: Tuple{Float64, Float64, Float64}
    if θ > ε_θ
        reciprocal_θ = 1 / θ
        s, c = sincos(θ)
        s2 = sin(2θ)
        α = reciprocal_θ + s2 * (reciprocal_θ ^ 2) / 2. - 2 * (s ^ 2) * (reciprocal_θ ^ 3)
        β = 2 * ((1 + c ^ 2) * (reciprocal_θ ^ 2) - s2 * (reciprocal_θ ^ 3))
        γ = 4 * (s * (reciprocal_θ ^ 3) - c * (reciprocal_θ ^ 2))
        return α, β, γ
    else
        θ2, θ4, θ6 = θ ^ 2, θ ^ 4, θ ^ 6
        α = (2. / 45) * θ ^ 3 - (2. / 315) * θ ^ 5 + (2. / 4725) * θ ^ 7
        β = (2. / 3) + (2. / 15) * θ2 - (4. / 105) * θ4 + (2. / 567) * θ6 - (4. / 22275) * θ ^ 8
        γ = 4. / 3 - (2. / 15) * θ2 + (1 / 210) * θ4 - (1. / 11340) * θ6
        return α, β, γ
    end
end

nodes(a:: Real, b:: Real, p:: Int) = range(a, b, 2p + 1)

function sample!(f, β:: Float64, γ:: Float64, nodes:: AbstractVector{Float64}, buffer:: Vector{Float64}):: AbstractVector{Float64}
    M = length(nodes)
    @inbounds for (j̃, node) in enumerate(nodes)
        j̃ = j - 1
        if iseven(j̃)
            buffer[j̃] = β * f(node)
        else
            buffer[j̃] = γ * f(node)
        end
    end
    return @views buffer[1:M] 
end

sample!(f, a:: Real, b:: Real, p:: Int, β:: Float64, γ:: Float64,  buffer:: Vector{Float64}) = sample!(f, β, γ, nodes(a, b, p), buffer)

function dft_portion(v:: Vector{Float64}, Δk:: Float64, a:: Float64, 
                     b:: Float64, p:: Int, 
                     fractional_fft_buffer:: FractionalFFTPlan{M}) where M
    @assert M == 2p + 1 "The buffer length must be equal 2p + 1 where p is the number 
                         of points in the Filon approximation"
    Δx = (b-a) / 2p
    δ = Δx * Δk / 2π
end

const Complex0:: Complex = zero(ComplexF64)

function E_2p_constant(fa:: Float64, fb:: Float64, κ:: Float64):: ComplexF64
    return (1. / 2) * (fa + fb * cis(-κ))
end

E_2pm1(odd_nodes:: AbstractVector{Float64}, v:: AbstractVector{Float64}, κ:: Float64) = dft(odd_nodes, v, κ)

function E(f, a:: Float64, b:: Float64, γ_k:: Float64, p::Int, buffer:: Vector{Float64}):: ComplexF64
    _nodes = nodes(a, b, p)
    α, β, γ = αβγ(h * κ)
    v = sample!(f, β, γ, _nodes, buffer)
    _dft = 
    fa, fb = f(a), f(b)
    _E_2p = E_2p(evens, buffer.even_buffer, κ)
    _E_2pm1 = E_2pm1(odds, buffer.odd_buffer, κ)
    h = (b-a) / (2p)
    return h * ((-im * α) * (buffer.even_buffer[1] - buffer.even_buffer[p + 1] * cis(-κ)) + β * _E_2p + γ * _E_2pm1)
end
end