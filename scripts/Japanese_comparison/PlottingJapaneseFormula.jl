include("./JapaneseFormula.jl")

using CairoMakie
using LaTeXStrings

function japanese_Π_plot(t1:: Real, t2:: Real, ks:: AbstractVector{<:Real},
    G_star:: Real; 
    plot_single:: Bool = true,
    plot_double:: Bool = true,
    atol:: Real = 1e-13, 
    rtol:: Real = 1e-3)

    pi_single = zeros(length(ks))
    pi_double = zeros(length(ks))
    pi_total  = zeros(length(ks))

    for (i, k) in enumerate(ks)
        s = compute_Pi_single(t1, t2, k; Gamma_star=G_star, atol=atol, rtol=rtol)
        d = compute_Pi_double(t1, t2, k; Gamma_star=G_star, atol=atol, rtol=rtol)
        
        pi_single[i] = abs(s)
        pi_double[i] = abs(d)
        pi_total[i]  = abs(s + d)
    end

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "k",
        ylabel = L"|\Pi(t_1, t_2, k)|",
        title = "Unequal-time Correlator (t1=$t1, t2=$t2, G*=$G_star)",
        xminorticksvisible = true, 
        yminorticksvisible = true,
        xminorgridvisible = true, 
        yminorgridvisible = true
    )

    if plot_single
        lines!(ax, ks, pi_single, label = "Single Bubble", color = :red, linewidth = 2, linestyle = :dash)
    end
    if plot_double
        lines!(ax, ks, pi_double, label = "Double Bubble", color = :blue, linewidth = 2, linestyle = :dash)
    end
    lines!(ax, ks, pi_total, label = L"Total |\Pi|", color = :black, linewidth = 2)
    return fig, ax
end
