# Figures for the demo. Two path panels share one set of axes.

using Plots
using Printf

function save_figures(directory, name, graph, comparison; subtitle = "")
    graph.truth === nothing && error("figures need a ground-truth trajectory")
    mkpath(directory)
    xlimits, ylimits = _limits(graph, comparison)
    colors = palette(:default)
    left = _paths(graph, comparison.dead_reckoning;
        title = "Dead reckoning", color = colors[1], xlimits, ylimits)
    right = _paths(graph, comparison.spectral;
        title = "Spectral", color = colors[2], xlimits, ylimits)
    note = "Dashed is the start. Solid is after LM. Black is ground truth. Gray is the loop closure."
    paths = plot(left, right;
        layout = (1, 2),
        size = (1000, 560),
        plot_title = subtitle * "\n" * note,
        plot_titlefontsize = 11,
        plot_titlevspan = 0.12,
    )
    cost = _cost(comparison, subtitle)
    paths_file = joinpath(directory, name * "_paths.png")
    cost_file = joinpath(directory, name * "_cost.png")
    savefig(paths, paths_file)
    savefig(cost, cost_file)
    return paths_file, cost_file
end

function _limits(graph, comparison)
    xs = Float64[]
    ys = Float64[]
    for poses in (
        graph.truth,
        comparison.dead_reckoning.init_poses,
        comparison.dead_reckoning.poses,
        comparison.spectral.init_poses,
        comparison.spectral.poses,
    )
        append!(xs, (pose.x for pose in poses))
        append!(ys, (pose.y for pose in poses))
    end
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    return (xmin - 0.08dx, xmax + 0.08dx), (ymin - 0.08dy, ymax + 0.08dy)
end

function _paths(graph, result; title, color, xlimits, ylimits)
    plt = plot(;
        aspect_ratio = :equal,
        legend = false,
        title = title,
        xlims = xlimits,
        ylims = ylimits,
        xlabel = _caption(result, graph.truth),
        ylabel = "",
    )
    _background!(plt, graph)
    ix, iy = _xy(result.init_poses)
    tx, ty = _xy(graph.truth)
    px, py = _xy(result.poses)
    plot!(plt, ix, iy; color, linestyle = :dash, linewidth = 1.5, label = "")
    plot!(plt, tx, ty; color = :black, linewidth = 2, label = "")
    plot!(plt, px, py; color, linewidth = 2.2, label = "")
    start = graph.truth[1]
    scatter!(plt, [start.x], [start.y]; color = :black, markersize = 3, label = "")
    return plt
end

function _background!(plt, graph)
    xs = Float64[]
    ys = Float64[]
    for edge in graph.edges
        edge.kind == :loop || continue
        a = graph.truth[edge.i]
        b = graph.truth[edge.j]
        append!(xs, (a.x, b.x, NaN))
        append!(ys, (a.y, b.y, NaN))
    end
    isempty(xs) || plot!(plt, xs, ys; color = :gray, linestyle = :dash, linewidth = 1, label = "")
    return nothing
end

_xy(poses) = ([pose.x for pose in poses], [pose.y for pose in poses])

function _caption(result, truth)
    err = pose_error(result.poses, truth)
    return @sprintf("RMSE %.3g,  cost %.3g,  %d iterations", err.pos_rmse, result.cost, result.iterations)
end

function _cost(comparison, subtitle)
    histories = (
        ("dead reckoning", comparison.dead_reckoning.cost_history),
        ("spectral", comparison.spectral.cost_history),
    )
    shown = [_shown_cost(history) for (_, history) in histories]
    positive = filter(>(0), vcat(shown...))
    use_log = !isempty(positive) && maximum(vcat(shown...)) / minimum(positive) > 80
    plt = plot(;
        xlabel = "iteration",
        ylabel = "cost",
        title = isempty(subtitle) ? "cost versus iteration" : subtitle,
        yscale = use_log ? :log10 : :identity,
        legend = :topright,
        size = (720, 440),
    )
    all_zero = all(iszero, vcat(shown...))
    if all_zero
        xlims!(plt, -0.5, 0.5)
        ylims!(plt, 0, 1)
    end
    for (k, (name, _)) in enumerate(histories)
        y = shown[k]
        plot!(plt, 0:length(y)-1, y;
            label = name,
            linewidth = 2,
            marker = length(y) <= 12 ? :circle : :none,
            markersize = 3,
        )
    end
    return plt
end

_shown_cost(history) = maximum(history) < 1e-8 ? zeros(length(history)) : history
