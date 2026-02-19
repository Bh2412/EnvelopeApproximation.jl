begin
    using EnvelopeApproximation
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.BubblesEvolution
    using EnvelopeApproximation.Spaces
    using EnvelopeApproximation.BoundaryConditions
    import EnvelopeApproximation.BubblesEvolution: BallSpace, sample_PT
    using StableRNGs
    using EnvelopeApproximation.EnvelopeAnalysis: align_ẑ
    using EnvelopeApproximation.GravitationalWaves
    using StaticArrays
    using LinearAlgebra
    using CairoMakie
    using Random
    using JLD2
    import CairoMakie as CM
    using Statistics
    using Printf
    using Dates
    using Measurements
end

begin  # Setup
    β = 1.
    Γ_0 = 1e-4
    t_0 = 0.
    tvp = 1 - 1e-5
    v_wall = 1.
    egp = ExponentialGrowthProcess(β, Γ_0, tvp; t_0=t_0, v_wall=v_wall)
    L = 80. / β
    center = EnvelopeApproximation.BubbleBasics.Point3(0., 0., 0.)
    space = BoxSpace(L, center)
    padding=0.
    ΔV = 1.
    
    sample_PT(rng:: StableRNG) = sample_PT(rng, egp, space, Periodic(); padding=padding)
    sample_PT(seed:: Int) = sample_PT(StableRNG(seed))
end

begin
    t1 = t2 = 3.
    ks = logrange(β / 10., (10 ^ (1//2)) * β, 100)
    initial_seed = 5122
    N = Threads.nthreads() * 1
    seeds = initial_seed:(initial_seed + N - 1)
    N_samples = 10_000_000
    res_lock = ReentrantLock()
    Π_res = zeros(Measurement{Float64}, length(ks))
    
    @time Threads.@threads for seed in seeds
        println("Thread $(Threads.threadid()): Computing for seed $seed...")
        rng = StableRNG(seed)
        strategy = MonteCarloSampling(N=N_samples, rng=rng)
        snapshot = sample_PT(rng)
        Π_sample = Π(t1, t2, ks, snapshot, space, Periodic(), strategy; ΔV=ΔV)
        lock(res_lock) do
            Π_res .+= Π_sample 
        end
    end
    Π_res ./= N
end

begin
    include("Japanese_comparison/PlottingJapaneseFormula.jl")
    fig, ax = japanese_Π_plot(t1, t2, ks, Γ_0; atol=1e-13, rtol=1e-3)
    vals, errs = Measurements.value.(Π_res), Measurements.uncertainty.(Π_res)
    mask = ((abs.(vals) .- errs) .> 0) .& (vals .> 0.)
    CM.scatter!(ax, ks[mask], vals[mask]; color=:red, label="Monte Carlo")
    CM.errorbars!(ax, ks[mask], abs.(vals[mask]), errs[mask]; color=:red)
    axislegend(ax, position = :lb)
    # save("BigPiComparisonToJapanese.png", fig)
    fig
end