using EnvelopeApproximation
using EnvelopeApproximation.BubbleBasics
using EnvelopeApproximation.BubblesEvolution
using EnvelopeApproximation.GeometricStressEnergyTensor
import EnvelopeApproximation.ChebyshevCFT: First3MomentsChebyshevPlan
import EnvelopeApproximation.GeometricStressEnergyTensor: Δ, k̂ik̂j∂_iφ∂_jφ, k̂ik̂jTij
import LinearAlgebra: norm
using CairoMakie
using BenchmarkTools
using HCubature
using StaticArrays
using EnvelopeApproximation.ISWPowerSpectrum
import EnvelopeApproximation.ISWPowerSpectrum: align_ẑ
using CSV

begin 
    R = 4.79
    d = 1.2
    nucleations = [(time=0., site=Point3(0., 0., -d / 2)), (time=0., site=Point3(0., 0., d / 2))]
    snapshot = BubblesSnapShot(nucleations, R)
    bubbles = current_bubbles(snapshot)
    k_0 = 2π / (R + d / 2)
    ks = logrange(k_0 / 1000, k_0 * 10, 2_000)
end

begin
    ΔV = 1.
    k̂ik̂j∂_iφ∂_jφ(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV)
end

begin
    bubbles = current_bubbles(snapshot)
    figure = Figure()
    ax = Axis(figure[1, 1], xlabel="k", ylabel="P(k)", 
              xscale=log10,
              yticks=0:50:150,
              limits = ((min(ks...), max(ks...)), (0, 200)),
              title="2 Bubble Surface-Surface, R=$R, d=$d")
    lines!(ax, ks, k̂ik̂j∂_iφ∂_jφ(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV) .|> abs ,label=L"T_{zz}")
    lines!(ax, ks, k̂ik̂j∂_iφ∂_jφ(ks, current_bubbles(align_ẑ(Vec3(1., 0., 0.)) * snapshot), chebyshev_plan, _Δ; ΔV=ΔV) .|> abs, label=L"T_{xx}")
    lines!(ax, ks, k̂ik̂j∂_iφ∂_jφ(ks, current_bubbles(align_ẑ(Vec3(0., 1., 0.)) * snapshot), chebyshev_plan, _Δ; ΔV=ΔV) .|> abs, label=L"T_{yy}", linestyle=:dash)
    # lines!(ax, ks, k̂ik̂j∂_iφ∂_jφ(ks, bubbles, chebyshev_plan, _Δ; ΔV=ΔV) .|> abs, label=L"$(\frac{\partial \varphi}{\partial z})^2$")
    axislegend(ax)
    save("/home/ben/Pictures/2_bubble_Tzz.png", figure)
    figure
end

begin
    chebyshev_plan = First3MomentsChebyshevPlan{32}()
    _Δ = Δ(4)
    # @profview for _ in 1:10 surface_P(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-3) end
    @time sp = surface_P(ks, snapshot, chebyshev_plan, _Δ; rtol=1e-3)
    x̂ = Vec3(1., 0., 0.)
    ŷ = Vec3(0., 1., 0.)
    @time xsp = surface_P(ks, align_ẑ(x̂) * snapshot, chebyshev_plan, _Δ; rtol=1e-3)
    @time ysp = surface_P(ks, align_ẑ(ŷ) * snapshot, chebyshev_plan, _Δ; rtol=1e-3)
end

begin
    μ = rand() * 2 - 1 
    ϕ = rand() * 2π
    random_n̂ = Vec3(sqrt(1 - μ ^ 2) * cos(ϕ), sqrt(1 - μ ^ 2) * sin(ϕ), μ)
    @show random_n̂
    @time n̂sp = surface_P(ks, align_ẑ(random_n̂) * snapshot, chebyshev_plan, _Δ; rtol=1e-3)
end

begin 
    figure = Figure()
    ax = Axis(figure[1, 1], xlabel="k", ylabel="P(k)", 
              xscale=log10, yscale=log10,
              title="2 Bubble Surface-Surface, R=$R, d=$d")
    lines!(ax, ks, sp, label="ẑ")
    # lines!(ax, ks, xsp, label="x̂")
    # lines!(ax, ks, ysp, linestyle=:dash, label="ŷ")
    lines!(ax, ks, n̂sp, linestyle=:dash, color="red", label="μ=$μ, ϕ=$ϕ")
    # scatter!(ax, ks, n̂sp, color=(:blue, 0.1))
    axislegend(ax, position=:lb)
    save("2_bubble_surface_surface.svg", figure)
    figure
end

begin 
    figure = Figure()
    ax = Axis(figure[1, 1], xlabel="k", ylabel="P(k)", 
              xscale=log10, yscale=log10,
              title="2 Bubble Surface-Surface, R=$R, d=$d")
    lines!(ax, ks, sp)
    save("2_bubble_surface_surface.png", figure)
    figure
end
