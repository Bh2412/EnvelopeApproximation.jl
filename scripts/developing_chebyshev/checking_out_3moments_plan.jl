using ApproxFun
using Plots
import EnvelopeApproximation.ChebyshevCFT: First3MomentsChebyshevPlan, values!, chebyshev_coeffs!, fourier_mode
using QuadGK
using BenchmarkTools

e0(x) = exp(x)
ẽ0(k, a=-1., b=1.; kwargs...) = quadgk(z -> e0(z) * cis(-k * z), a, b; kwargs...)
e1(x) = x * exp(x)
ẽ1(k, a=-1., b=1.; kwargs...) = quadgk(z -> e1(z) * cis(-k * z), a, b; kwargs...)
e2(x) = x^2 * exp(x)
ẽ2(k, a=-1., b=1.; kwargs...) = quadgk(z -> e2(z) * cis(-k * z), a, b; kwargs...)

plan = First3MomentsChebyshevPlan{32}()

values!(e0, -1., 1., plan)
chebyshev_coeffs!(e0, -1., 1., plan)

x = -1. : 0.001: 1.

# plot(x, @. e0.(x) * sqrt(1 - x ^ 2))
# plot!(x, Fun(Chebyshev(), plan.coeffs0_buffer).(x), label="Approximation")

# plot(x, (@. e1.(x) * sqrt(1 - x^2)), label="exact")
# plot!(x, Fun(Chebyshev(), plan.coeffs1_buffer).(x), label="Approximation")

# plot(x, (@. e2.(x) * sqrt(1 - x^2)), label="exact")
# plot!(x, Fun(Chebyshev(), plan.coeffs2_buffer).(x), label="Approximation")

a, b = -1., 1.
# @btime chebyshev_coeffs!($e0, $a, $b, $plan)
# @profview for _ in 1:10_000 chebyshev_coeffs!(e0, a, b, plan) end

chebyshev_coeffs!(e0, -1., 1., plan)
scale, translation = (b - a) / 2, (a + b) / 2
ks = range(0., 10., 1_000)
modes = fourier_mode.(ks, (plan, ), (scale, ), (translation, ))
modes0 = modes .|> x -> x[1]
modes1 = modes .|> x -> x[2]
modes2 = modes .|> x -> x[3]

# plot(ks, ẽ0.(ks) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes0 .|> real, label="approximation")

# plot(ks, ẽ0.(ks) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes0 .|> imag, label="approximation")

# plot(ks, ẽ1.(ks) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes1 .|> real, label="approximation")

# plot(ks, ẽ1.(ks) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes1 .|> imag, label="approximation")

# plot(ks, ẽ2.(ks) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes2 .|> real, label="approximation")

# plot(ks, ẽ2.(ks) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes2 .|> imag, label="approximation")

a, b = 1., 2.
chebyshev_coeffs!(e0, a, b, plan)
scale, translation = (b - a) / 2, (a + b) / 2
ks = range(0., 10., 1_000)
modes = fourier_mode.(ks, (plan, ), (scale, ), (translation, ))
modes0 = modes .|> x -> x[1]
modes1 = modes .|> x -> x[2]
modes2 = modes .|> x -> x[3]


# plot(ks, ẽ0.(ks, (a, ), (b, )) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes0 .|> real, label="approximation")

# plot(ks, ẽ0.(ks, (a, ), (b, )) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes0 .|> imag, label="approximation")

# plot(ks, ẽ1.(ks, (a, ), (b, )) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes1 .|> real, label="approximation")

# plot(ks, ẽ1.(ks, (a, ), (b, )) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes1 .|> imag, label="approximation")

# plot(ks, ẽ2.(ks, (a, ), (b, )) .|> x -> real(x[1]), label="exact")
# plot!(ks, modes2 .|> real, label="approximation")

# plot(ks, ẽ2.(ks, (a, ), (b, )) .|> x -> imag(x[1]), label="exact")
# plot!(ks, modes2 .|> imag, label="approximation")

k = 1.
# @btime fourier_mode($k, $plan, $scale, $translation)
# @profview for _ in 1:100_000 fourier_mode(k, plan, scale, translation) end
