using Intervals
using FastTransforms
using Bessels
import Bessels: besselj!
using QuadGK
using ApproxFun
using BenchmarkTools
using Plots
import LinearAlgebra.mul!

const ChebyshevInterval:: Intervals.Interval = Intervals.Interval(-1. ,1.)

struct ChebyshevPlan{N}
    points:: Vector{Float64}
    coeffs_buffer:: Vector{Float64}
    bessels_buffer:: Vector{Float64}
    transform_plan:: FastTransforms.ChebyshevTransformPlan{Float64, 1, Vector{Int32}, true, 1, Tuple{Int64}}

    function ChebyshevPlan{N}() where N
        points = chebyshevpoints(Float64, N, Val(1))
        coeffs_buffer = Vector{Float64}(undef, N)
        bessels_buffer = Vector{Float64}(undef, N)
        transform_plan = plan_chebyshevtransform!(zeros(N), Val(1))
        return new{N}(points, coeffs_buffer, bessels_buffer, transform_plan)
    end
end

translation(a:: Float64, b:: Float64):: Float64 = (a + b) / 2
scale(a:: Float64, b:: Float64):: Float64 = (b - a) / 2
u(x:: Float64, scale:: Float64, translation:: Float64):: Float64 = (x - translation) / scale
inverse_u(u:: Float64, scale:: Float64, translation:: Float64):: Float64 = translation + scale * u
inverse_chebyshev_weight(u:: Float64):: Float64 = sqrt(1 - u ^ 2)

function values!(f, a:: Float64, b:: Float64, 
                 chebyshev_plan:: ChebyshevPlan{N}):: AbstractVector{Float64} where N
    scale_factor = scale(a, b)
    t = translation(a, b)
    for (i, u) in enumerate(chebyshev_plan.points)
        chebyshev_plan.coeffs_buffer[i] = f(inverse_u(u, scale_factor, t)) * inverse_chebyshev_weight(u) 
    end     
    return chebyshev_plan.coeffs_buffer
end

function chebyshev_coeffs!(f, a:: Float64, b:: Float64, chebyshev_plan:: ChebyshevPlan{N}):: Vector{Float64} where N
    return chebyshev_plan.transform_plan * values!(f, a, b, chebyshev_plan)
end

# Followins equation 2.10.5.1 and 2.10.5.2 in "methods of numerical integration"

function fourier_mode(k:: Float64, 
                      chebyshev_plan:: ChebyshevPlan{N}, 
                      scale:: Float64 = 1.,
                      translation:: Float64 = 0.):: ComplexF64 where N
    k̃ = scale * k
    besselj!(chebyshev_plan.bessels_buffer, 0:(N-1), k̃)
    c, s = 0., 0.
    @inbounds for ((ĩ, coeff), bessel_value) in zip(enumerate(chebyshev_plan.coeffs_buffer), chebyshev_plan.bessels_buffer)
        i = ĩ - 1
        l = i ÷ 2
        if iseven(i)
            c += ((-1) ^ l) * π * coeff * bessel_value
        else
            s += ((-1) ^ l) * π * coeff * bessel_value
        end
    end
    return (c - im * s) * cis(-k * translation) * scale
end

function fourier_modes!(ks:: AbstractVector{Float64}, 
                        f, a:: Float64, b:: Float64, 
                        chebyshev_plan:: ChebyshevPlan{N}, 
                        buffer:: Vector{ComplexF64}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    @inbounds for (j, k) in enumerate(ks)
        buffer[j] = fourier_mode(k, chebyshev_plan, a, t)
    end
    return buffer
end

function add_fourier_modes!(ks:: AbstractVector{Float64}, 
                            f, a:: Float64, b:: Float64, 
                            chebyshev_plan:: ChebyshevPlan{N}, 
                            buffer:: Vector{ComplexF64}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    @inbounds for (j, k) in enumerate(ks)
        buffer[j] += fourier_mode(k, chebyshev_plan, a, t)
    end
end


function fourier_modes(ks:: AbstractVector{Float64}, 
                       f, a:: Float64, b:: Float64, chebyshev_plan:: ChebyshevPlan{N}) where N
    chebyshev_coeffs!(f, a, b, chebyshev_plan)
    t = translation(a, b)
    a = scale(a, b)
    return fourier_mode.(ks, (chebyshev_plan, ), (a, ), (t, ))
end

function fourier(g, k:: Float64, a:: Float64, b:: Float64; 
                 kwargs...)
    return quadgk(x -> g(x) * cis(-k * x), a, b; kwargs...)
end

n = 2^5
chebyshev_plan = ChebyshevPlan{n}()
a, b = -1., 1.
exp_coeffs = chebyshev_coeffs!(exp, a, b, chebyshev_plan)
exp_approximation = x -> [cos(k*acos(x)) for k=0:n-1]' * exp_coeffs
x = a:0.01:b
plot(x, [exp(z) * sqrt(1 - z ^2) for z in x], label = "exp(x)")
plot!(x, exp_approximation.(x), label = "exp(x) approximation")

# Example

a, b = 1., 2.
n = 2 ^ 5
plan = ChebyshevPlan{n}()
ks = 0.01:0.01:10
new_method_modes = fourier_modes(ks, sin, a, b, plan)
quadgk_modes = fourier.((sin, ), ks, (a, ), (b, )) .|>  x -> x[1]

(new_method_modes - quadgk_modes) .|> abs |> x -> max(x...)


sin_coeffs = chebyshev_coeffs!(sin, a, b, plan)
# @btime sin_coeffs = chebyshev_coeffs!(sin, a, b, n, V)
scale_factor = scale(a, b)
t = translation(a, b)
k = ks[end]
# @btime fourier_mode($k, $plan, $scale_factor, $t)
# @profview for _ in 1:100_000 fourier_mode(k, plan, scale_factor, t) end


ks = range(0.1, 10., 1000)
# @btime fourier_modes($ks, $sin, $a, $b, $plan)
g(k) = quadgk(x -> sin(x) * cis(-k * x), a, b)
# @btime g.(ks)

using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.GeometricStressEnergyTensor: intersection_domes, BubbleArck̂ik̂j∂iφ∂jφ, ∫_ϕ
import EnvelopeApproximation.GeometricStressEnergyTensor: align_ẑ, polar_limits, ring_dome_intersection
import EnvelopeApproximation.GeometricStressEnergyTensor: fourier_mode
import EnvelopeApproximation.GeometricStressEnergyTensor: intersection_domes, BubbleArck̂ik̂j∂iφ∂jφ, ∫_ϕ
import EnvelopeApproximation.GeometricStressEnergyTensor: align_ẑ, polar_limits, ring_dome_intersection
import EnvelopeApproximation.GeometricStressEnergyTensor: fourier_mode
import EnvelopeApproximation.GeometricStressEnergyTensor: _buffers, ring_domes_intersection!
import LinearAlgebra: norm
#using Intervals

R = 2.
d = 2.4
nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
snapshot = BubblesSnapShot(nucleations, R)
bubbles = current_bubbles(snapshot)
k_0 = 2π / (R + d / 2)
ks = LinRange(k_0 / 10, k_0 * 10, 100)
k_vecs = (x -> Vec3(0., 0., x)).(ks)
k_0
ks
ts = LinRange(0., R, 10) |> collect
η_PT = R

bubble = bubbles[1]
domes = intersection_domes(bubbles)
bubble_domes = domes[1]

struct Δ
    arcs_buffer:: Vector{PeriodicInterval}
    limits_buffer:: Vector{Tuple{Float64, Float64}}
    intersection_buffer:: Vector{PeriodicInterval}
end

Δ(n:: Int64) = Δ(_buffers(n)...)

function (δ:: Δ)(μ:: Float64, bubble:: Bubble, 
                 intersection_domes:: Vector{IntersectionDome}):: Float64
    periodic_intervals = ring_domes_intersection!(μ, bubble.radius, intersection_domes, 
                                                  δ.arcs_buffer, δ.limits_buffer, δ.intersection_buffer)
    return sum((p.Δ for p in periodic_intervals), init=0.)
end

const _Δ:: Δ = Δ(5)
# const b:: Bubble = bubble
const bdomes:: Vector{IntersectionDome} = bubble_domes


function bubble_Δ(μ:: Float64) :: Float64
    return _Δ(μ, bubble, bdomes)
end

bubble_Δ(0.8)
μ_limits = polar_limits(bubble.radius, bdomes)[[1, 2]]

f(μ) = μ ^ 2 * bubble_Δ(μ)
f̃(κ) = quadgk(μ -> f(μ) * cis(-κ * μ),  μ_limits[1], μ_limits[2])[1]

n = 2^5
chebyshev_plan = ChebyshevPlan{n}()
a, b = μ_limits
ks = range(0.01, 100., 100)
@btime $chebyshev_coeffs!($f, $a, $b, $chebyshev_plan)
fourier_modes(ks, f, a, b, chebyshev_plan) - f̃.(ks)
plot(ks, fourier_modes(ks, f, a, b, chebyshev_plan) .|> real)
plot!(ks, f̃.(ks) .|> real)
@btime $fourier_modes($ks, $f, $a, $b, $chebyshev_plan) 
@btime $f̃.($ks)
# @profview for _ in 1:10_000 f̃.(ks) end