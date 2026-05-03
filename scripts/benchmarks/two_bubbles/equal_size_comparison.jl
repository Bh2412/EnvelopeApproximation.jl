begin
    using Printf
    import CairoMakie as CM
    using EnvelopeApproximation.BubbleBasics
    using EnvelopeApproximation.StressEnergyTensorContractions
    using LinearAlgebra

    include("Analytic.jl")
    include("Numeric.jl")
end

# --- Configuration ---
begin
    r     = 5.0
    d     = 6.0
    ρ_vac = 1.0
    L     = 6 * (r + d / 2)
    V     = L^3

    ks = collect(LinRange(0.1, 2.0, 100))

    # Four directions in the xz plane parameterised by cos(ξ)
    cos_ξs = [0., 1/3, 2/3, 1.]
    k̂s     = [normalize([sqrt(max(1 - cξ^2, 0.)), 0., cξ]) for cξ in cos_ξs]
    colors  = [:steelblue, :crimson, :forestgreen, :darkorange]

    println("r=$r  d=$d  L=$L  $(length(ks)) k-points  $(length(cos_ξs)) directions")
end

# --- Compute ---
begin
    println("Computing numeric (T_ij azimuthal reduction)...")
    numerics = [NumericTwoPointTij(k̂,  ks, r, r, d, ρ_vac, V) for k̂ in k̂s]

    println("Computing analytic...")
    analytics = [[AnalyticTwoPointTij(k .* k̂s[di], r, r, d, ρ_vac, V) for k in ks]
                 for di in eachindex(cos_ξs)]
end

# --- Coefficient-level agreement (max rel error per direction) ---
begin
    println("\nCoefficient agreement (max rel error over all k and 6×6 entries):")
    threshold = 1e-14
    for (di, cξ) in enumerate(cos_ξs)
        max_rel = 0.0
        for ki in eachindex(ks)
            N = numerics[di][:, :, ki]
            A = Matrix(analytics[di][ki])
            max_rel = max(max_rel, maximum(
                abs(A[i]) > threshold ? abs(N[i] - A[i]) / abs(A[i]) : 0.0
                for i in CartesianIndices(A)
            ))
        end
        flag = max_rel > 1e-3 ? "  !" : ""
        @printf("  cos(ξ) = %.5f  →  max rel err = %.3e%s\n", cξ, max_rel, flag)
    end
end

# --- Contractions: numeric vs analytic for all directions ---
begin
    k_vecs     = [[k .* k̂s[di] for k in ks] for di in eachindex(cos_ξs)]
    ana_contrs = [[all_contractions(Matrix(analytics[di][ki]), k_vecs[di][ki]) for ki in eachindex(ks)]
                  for di in eachindex(cos_ξs)]
    num_contrs = [[all_contractions(numerics[di][:, :, ki],    k_vecs[di][ki]) for ki in eachindex(ks)]
                  for di in eachindex(cos_ξs)]

    # TT, T, σσ, pσ in a 2×2 top grid; pp spans the full bottom row
    top_names  = [:TT, :T, :σσ, :pσ]
    top_titles = ["Λ_TT (GW)", "Λ_T (vector)", "Λ_σσ (anis. stress)", "Λ_pσ (mixed)"]

    fig = CM.Figure(size = (1300, 900))

    function plot_direction!(ax, name)
        for (di, cξ) in enumerate(cos_ξs)
            c = colors[di]
            CM.lines!(ax,   ks, real.(getproperty.(ana_contrs[di], name)); color = c, linewidth = 2)
            CM.scatter!(ax, ks, real.(getproperty.(num_contrs[di], name)); color = c, marker = :x, markersize = 10)
        end
    end

    for (ci, (name, title)) in enumerate(zip(top_names, top_titles))
        row = (ci - 1) ÷ 2 + 1
        col = (ci - 1) % 2 + 1
        plot_direction!(CM.Axis(fig[row, col], title = title, xlabel = "k", ylabel = "Re(contraction)"), name)
    end

    plot_direction!(CM.Axis(fig[3, 1:2], title = "Λ_pp (pressure)", xlabel = "k", ylabel = "Re(contraction)"), :pp)

    dir_elems  = [[CM.LineElement(color = colors[di], linewidth = 2),
                   CM.MarkerElement(color = colors[di], marker = :x, markersize = 10)]
                  for di in eachindex(cos_ξs)]
    dir_labels = [@sprintf("cos(ξ) = %.5f", cξ) for cξ in cos_ξs]

    style_elems  = [[CM.LineElement(color = :black, linewidth = 2)],
                    [CM.MarkerElement(color = :black, marker = :x, markersize = 10)]]
    style_labels = ["analytic", "numeric"]

    CM.Legend(fig[1:2, 3],
        [dir_elems,   style_elems],
        [dir_labels,  style_labels],
        ["k̂ direction", "style"])

    CM.save("$(@__DIR__)/two_bubble_contractions.png", fig)
    println("Saved two_bubble_contractions.png")
    fig
end
