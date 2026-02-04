begin
    using EnvelopeApproximation
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.Spaces
    using EnvelopeApproximation.BoundaryConditions
    using EnvelopeApproximation.BubblesEvolution
    import EnvelopeApproximation.BubblesEvolution:sample_PT
    using StableRNGs
    using EnvelopeApproximation.EnvelopeAnalysis: align_ẑ
    # Import the physics kernel and the generic API
    using EnvelopeApproximation.GravitationalWaves: Π, Directional_Π, x̂_ix̂_j, Λx̂x̂
    using EnvelopeApproximation.SphericalHarmonics: SHPlan
    using EnvelopeApproximation.QuadGKCFT: VectorQuadGKPlan
    using StaticArrays
    using LinearAlgebra
    using CairoMakie
    using Random
    using JLD2
    using CairoMakie
    using Statistics
    using Printf
    using Dates
end

#=
Parameters taken from Kosowsky and Turner
=#

begin  # Setup
    β = 1.
    Γ_0 = 1e-4
    t_0 = 0.
    tvp = 1 - 1e-5
    v_wall = 1.
    egp = ExponentialGrowthProcess(β, Γ_0, tvp; t_0=t_0, v_wall=v_wall)
    sequential_egp = ExponentialGrowthProcess{SequentialSampling}(β, Γ_0, tvp; t_0=t_0, v_wall=v_wall)
    
    sample_PT(rng:: StableRNG) = sample_PT(rng, egp, box_space, Periodic(); padding=padding)
    sample_PT(seed:: Int) = sample_PT(StableRNG(seed))
end



# --- Main Execution ---
begin
    t1 = t2 = 5.
    direction = EnvelopeApproximation.BubbleBasics.Vec3(0., 0., 1.)
    ks = logrange(β / 10., 10 * β, 100)
    num_realizations = 8
    l_max = 64

    results_Π = zeros(ComplexF64, num_realizations, length(ks))
    
    println("Starting simulation of $num_realizations realizations...")
    println("Parameters: l_max=$l_max, t1=$t1, t2=$t2")

    Threads.@threads for i in 1:num_realizations
        seed = 1000 + i 
        println("Initiating seed $seed on Thread $(Threads.threadid())")
        
        # 1. Thread-local QuadGK Plan
        local_cft_plan = VectorQuadGKPlan{2}(; atol=1e-5, rtol=1e-3, initdiv=100)

        # 2. Physics Generation
        snapshot = sample_PT(seed)
        _Λx̂x̂ = Λx̂x̂(1_000)
        
        # 3. Integration Plan 
        # (Imported from SphericalHarmonics.jl)
        angular_integration_plan = SHPlan(l_max, length(ks))
        
        # 4. Compute Π
        # Note: Arguments match Source 115 in GravitationalWaves.jl
        val_Π = Π(t1, t2, ks, snapshot, box_space, Periodic(), local_cft_plan, angular_integration_plan, _Λx̂x̂)
        
        results_Π[i, :] = val_Π
        
        @lock Threads.SpinLock() begin
            mid_idx = div(length(ks), 2)
            @printf("Run %d/%d (Thread %d): Π[k≈%.2f] = %.2e\n", 
                    i, num_realizations, Threads.threadid(), ks[mid_idx], real(val_Π[mid_idx]))
        end
    end
end

begin
    """
        save_simulation_data(filename, results_Π, ks, params...)

    Saves the simulation results and metadata to a JLD2 file.
    Automatically appends a timestamp to the filename if not provided.
    """
    function save_simulation_data(base_filename::String, 
                                results_Π::Matrix{ComplexF64}, 
                                ks::Vector{Float64};
                                l_max::Int=0,
                                t_target::Float64=0.0)
        
        # Create a filename with timestamp to prevent overwriting
        timestamp = Dates.format(now(), "yyyy_mm_dd")
        filename = "$(base_filename)_$(timestamp).jld2"

        println("Saving data to $filename...")
        
        jldsave(filename; 
            results_Π = results_Π,
            ks = ks,
            l_max = l_max,
            t_target = t_target,
            timestamp = timestamp
        )
        
        println("Save complete.")
        return filename
    end

    save_simulation_data("KT_bigpi", results_Π, collect(ks), ;l_max=l_max, t_target=t_target)
   
end

begin
    """
        load_and_view_results(filename)

    Loads a JLD2 simulation file and generates a statistical plot 
    (Mean ± StdDev) of the anisotropic stress power Π.
    """
    function load_and_view_results(filename::String)
        println("Loading $filename...")
        
        # Load data
        data = load(filename)
        results_Π = data["results_Π"]
        ks = data["ks"]
        l_max = get(data, "l_max", "Unknown")
        t_target = get(data, "t_target", "Unknown")
        
        # Calculate Statistics
        # results_Π is (num_realizations x num_ks)
        real_Π = real.(results_Π)
        mean_Π = vec(mean(real_Π, dims=1))
        std_Πdev = vec(std(real_Π, dims=1))
        
        num_realizations = size(results_Π, 1)

        # --- Visualization ---
        f = Figure(size = (1000, 600))
        
        # Main Title
        Label(f[0, :], "Anisotropic Stress Π Source (l_max=$l_max, t=$t_target)", 
            fontsize = 20, font = :bold)

        # Axis Setup
        ax = Axis(f[1, 1], 
            xlabel = "Wavenumber k", 
            ylabel = "Re[Π]",
            xscale = log10,
            yscale=log10,
            title = "Power Spectrum Statistics (N=$num_realizations)",
            xgridstyle = :dash,
            ygridstyle = :dash
        )

        # 1. Plot individual realizations (faint background lines)
        for i in 1:num_realizations
            lines!(ax, ks, real_Π[i, :], color=(:grey, 0.3), linewidth=1)
        end

        # 2. Plot Error Band (Mean ± 1 Std Dev)
        band!(ax, ks, mean_Π .- std_Πdev, mean_Π .+ std_Πdev, 
            color = (:cornflowerblue, 0.4), label="±1σ Deviation")

        # 3. Plot Mean Line
        lines!(ax, ks, mean_Π, color = :blue, linewidth = 3, label = "Mean Π")
        scatter!(ax, ks, mean_Π, color = :blue, markersize = 8)

        # 4. Zero line reference
        hlines!(ax, [0.0], color=:black, linestyle=:dash, linewidth=1)

        # Legend
        axislegend(ax, position = :rt)

        # Print summary statistics to console
        println("\n--- Data Summary ---")
        println("Dimensions: $(size(results_Π))")
        println("k range:    [$(minimum(ks)), $(maximum(ks))]")
        println("Max Mean Π: $(maximum(mean_Π))")
        println("Min Mean Π: $(minimum(mean_Π))")

        return f
    end

    fig = load_and_view_results("/home/ben/dev/EnvelopeApproximation/KT_bigpi_2025_12_14.jld2")
end