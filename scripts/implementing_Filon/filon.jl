using BenchmarkTools
using QuadGK
using CairoMakie

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

struct FilonBuffer
    even_buffer:: Vector{Float64}
    odd_buffer:: Vector{Float64}
end

function FilonBuffer(n:: Int)
    even_buffer = Vector{Float64}(undef, n)
    odd_buffer = Vector{Float64}(undef, n)
    return FilonBuffer(even_buffer, odd_buffer)
end

even_nodes(a:: Real, b:: Real, p:: Int64) = range(a, b, p + 1)
function odd_nodes(a:: Real, b:: Real, p:: Int64)
    h = (b-a) / 2p
    return range(a + h, b - h, p)
end

a, b = 1., 2.
p = 10
# @btime even_nodes($a, $b, $p)
# @btime $odd_nodes($a, $b, $p)

function sample!(f, nodes:: AbstractVector{Float64}, buffer:: Vector{Float64})
    p = length(nodes)
    @inbounds for (i, node) in enumerate(nodes)
        buffer[i] = f(node)
    end
    @views buffer[1:p] 
end

filoner = FilonBuffer(1000)
nodes = even_nodes(a, b, p)
# @btime $sample!($sin, $nodes, $filoner.even_buffer)
# @profview for _ in 1:1_000_000 sample!(sin, nodes, filoner.even_buffer) end

function sample!(f, a:: Float64, b:: Float64, p:: Int64, buffer:: FilonBuffer)
    evens = even_nodes(a, b, p)
    odds = odd_nodes(a, b, p)
    sample!(f, evens, buffer.even_buffer)
    sample!(f, odds, buffer.odd_buffer)
end

function sample!(f, evens:: AbstractVector{Float64}, odds:: AbstractVector{Float64}, buffer:: FilonBuffer)
    sample!(f, evens, buffer.even_buffer)
    sample!(f, odds, buffer.odd_buffer)
end

const Complex0:: Complex = Complex(0.)

function dft(nodes:: AbstractVector{Float64}, v:: AbstractVector{Float64}, κ:: Float64):: ComplexF64
    return sum((_v * cis(-κ * node) for (node, _v) in zip(nodes, v)), init=Complex0)
end

function E_2p(even_nodes:: AbstractVector{Float64}, even_values:: Vector{Float64}, κ:: Float64):: ComplexF64
    _dft = dft(even_nodes, even_values, κ)
    return _dft - (1. / 2) * (even_values[1] + even_values[end] * cis(-κ))
end

E_2pm1(odd_nodes:: AbstractVector{Float64}, v:: AbstractVector{Float64}, κ:: Float64) = dft(odd_nodes, v, κ)

function E(f, a:: Float64, b:: Float64, κ:: Float64, p::Int, buffer:: FilonBuffer):: ComplexF64
    evens, odds = even_nodes(a, b, p), odd_nodes(a, b, p)
    sample!(f, evens, odds, buffer)
    _E_2p = E_2p(evens, buffer.even_buffer, κ)
    _E_2pm1 = E_2pm1(odds, buffer.odd_buffer, κ)
    h = (b-a) / (2p)
    α, β, γ = αβγ(h * κ)
    return h * ((-im * α) * (buffer.even_buffer[1] - buffer.even_buffer[p + 1] * cis(-κ)) + β * _E_2p + γ * _E_2pm1)
end

a, b, κ, p = 1., 2., 10., 30
filon_buffer = FilonBuffer(10_000)
# @btime $E($sin, $a, $b, $κ, $p, $filon_buffer)
# @btime quadgk_count(x -> sin(x) * cis(-κ * x), $a, $b; rtol=1e-2)
# @profview for _ in 1:1000_000 quadgk_count(x -> sin(x) * cis(-κ * x), a, b; rtol=1e-2) end

_E(κ) = E(sin, a, b, κ, p, filon_buffer)
_exact_sin_int(κ) = quadgk(x -> sin(x) * cis(-κ * x), a, b)

κs = 0.: 0.1: 100.
new_method = _E.(κs)
exact = _exact_sin_int.(κs) .|> x -> x[1]

begin
    fig = Figure()
    axr = Axis(fig[1, 1])
    lines!(axr, κs, new_method .|> real, label="approximation")
    lines!(axr, κs, exact .|> real, label="exact")
    axi = Axis(fig[1, 2])
    lines!(axi, κs, new_method .|> imag, label="approximation")
    lines!(axi, κs, exact .|> imag, label="exact")
    fig
end


# diff(κ) = _E(κ) - _exact_sin_int(κ)[1]

# lines(κs, diff.(κs) .|> abs)
# diff.(κs) .|> abs |> x -> max(x...)

# @btime $E($sin, $a, $b, $κ, $p, $filon_buffer)
# @profview for _ in 1:10_000 E(sin, a, b, κ, p, filon_buffer) end
