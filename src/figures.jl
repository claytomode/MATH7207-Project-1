"""Presentation figures, written as SVG so the demo has no plotting dependency."""

function _xml(text::AbstractString)
    return replace(replace(replace(text, "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")
end

struct _Map
    xmin::Float64
    ymin::Float64
    scale::Float64
    origin_x::Float64
    origin_y::Float64 # SVG y of the data point (xmin, ymin); the bottom-left of the axes
end

function _map(xmin, xmax, ymin, ymax, left, top, plot_w, plot_h)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    scale = min(plot_w / dx, plot_h / dy)
    used_w = scale * dx
    used_h = scale * dy
    origin_x = left + (plot_w - used_w) / 2
    # Bottom of the used rectangle sits on the bottom of the plot box.
    origin_y = top + (plot_h - used_h) / 2 + used_h
    return _Map(xmin, ymin, scale, origin_x, origin_y)
end

_sx(m::_Map, x) = m.origin_x + (x - m.xmin) * m.scale
_sy(m::_Map, y) = m.origin_y - (y - m.ymin) * m.scale

function _polyline(m::_Map, poses::AbstractVector{Pose})
    io = IOBuffer()
    for (k, pose) in enumerate(poses)
        k > 1 && print(io, " ")
        @printf(io, "%.2f,%.2f", _sx(m, pose.x), _sy(m, pose.y))
    end
    return String(take!(io))
end

"""
Write `<name>_paths.svg` and `<name>_cost.svg` into `directory`.

The path figure has two panels on one scale. Dead reckoning is on the left and
the spectral start is on the right. Both show the ground truth, the starting
guess, and the refined trajectory.
"""
function save_figures(directory::AbstractString, name::AbstractString, graph::PoseGraph, comparison::Comparison; subtitle::AbstractString = "")
    graph.truth === nothing && error("figures need a ground-truth trajectory")
    mkpath(directory)
    paths_file = joinpath(directory, name * "_paths.svg")
    cost_file = joinpath(directory, name * "_cost.svg")
    write(paths_file, _paths_svg(graph, comparison, subtitle))
    write(cost_file, _cost_svg(comparison, subtitle))
    return paths_file, cost_file
end

function _paths_svg(graph::PoseGraph, comparison::Comparison, subtitle::AbstractString)
    truth = graph.truth
    trajectories = (
        truth,
        comparison.dead_reckoning.init_poses,
        comparison.dead_reckoning.poses,
        comparison.spectral.init_poses,
        comparison.spectral.poses,
    )
    xs = Float64[]
    ys = Float64[]
    for poses in trajectories
        append!(xs, (p.x for p in poses))
        append!(ys, (p.y for p in poses))
    end
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy

    width = 1200
    height = 760
    panels = (
        (title = "Dead reckoning", result = comparison.dead_reckoning, color = "#a93226", left = 40.0),
        (title = "Spectral", result = comparison.spectral, color = "#1a5276", left = 630.0),
    )
    top = 108.0
    plot_w = 520.0
    plot_h = 520.0
    arrow = 0.045 * max(xmax - xmin, ymax - ymin)
    stride = max(1, length(truth) ÷ 12)

    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(io, """<rect width="100%" height="100%" fill="#ffffff"/>""")
    println(io, """<text x="40" y="42" font-family="Segoe UI, Helvetica, sans-serif" font-size="26" font-weight="600" fill="#1c1c1c">$(_xml(subtitle))</text>""")
    println(io, """<text x="40" y="72" font-family="Segoe UI, Helvetica, sans-serif" font-size="15" fill="#555555">Black is ground truth. Dashed is the start. Solid color is after Levenberg–Marquardt. Gray is the loop closure.</text>""")

    for panel in panels
        mapper = _map(xmin, xmax, ymin, ymax, panel.left, top, plot_w, plot_h)
        println(io, """<text x="$(panel.left)" y="100" font-family="Segoe UI, Helvetica, sans-serif" font-size="20" font-weight="600" fill="#1c1c1c">$(panel.title)</text>""")
        _draw_panel!(io, mapper, truth, panel.result, panel.color, arrow, stride, graph)
        err = pose_error(panel.result.poses, truth)
        caption = @sprintf("final cost %.3g   ·   %d iterations   ·   position RMSE %.3g", panel.result.cost, panel.result.iterations, err.pos_rmse)
        println(io, """<text x="$(panel.left)" y="660" font-family="Segoe UI, Helvetica, sans-serif" font-size="15" fill="#333333">$caption</text>""")
    end
    println(io, "</svg>")
    return String(take!(io))
end

function _draw_panel!(io, mapper, truth, result, color, arrow, stride, graph)
    for edge in graph.edges
        edge.kind == :loop || continue
        a = truth[edge.i]
        b = truth[edge.j]
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#b0b0b0\" stroke-width=\"1.6\" stroke-dasharray=\"3 4\"/>\n",
            _sx(mapper, a.x), _sy(mapper, a.y), _sx(mapper, b.x), _sy(mapper, b.y))
    end
    @printf(io, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"1.8\" stroke-dasharray=\"7 5\" points=\"%s\"/>\n",
        color, _polyline(mapper, result.init_poses))
    @printf(io, "<polyline fill=\"none\" stroke=\"#222222\" stroke-width=\"2.5\" points=\"%s\"/>\n", _polyline(mapper, truth))
    @printf(io, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"2.6\" points=\"%s\"/>\n", color, _polyline(mapper, result.poses))
    for (poses, stroke) in ((truth, "#222222"), (result.poses, color))
        for k in 1:stride:length(poses)
            pose = poses[k]
            x2 = pose.x + arrow * cos(pose.θ)
            y2 = pose.y + arrow * sin(pose.θ)
            @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"%s\" stroke-width=\"1.3\"/>\n",
                _sx(mapper, pose.x), _sy(mapper, pose.y), _sx(mapper, x2), _sy(mapper, y2), stroke)
        end
    end
    start = truth[1]
    @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"4.5\" fill=\"#222222\"/>\n", _sx(mapper, start.x), _sy(mapper, start.y))
end

function _axis_ticks(lo, hi; integer::Bool = false)
    span = hi - lo
    span <= 0 && return [lo]
    raw = span / 4
    pow = 10.0^floor(log10(raw))
    step = raw / pow < 1.5 ? pow : (raw / pow < 3.5 ? 2pow : (raw / pow < 7.5 ? 5pow : 10pow))
    integer && (step = max(step, 1.0))
    start = ceil(lo / step) * step
    ticks = Float64[]
    value = start
    while value <= hi + 1e-9 * step && length(ticks) < 8
        push!(ticks, value)
        value += step
    end
    return isempty(ticks) ? [lo, hi] : ticks
end

function _log_ticks(lo, hi)
    lo <= 0 && return Float64[]
    ticks = Float64[]
    decade = floor(Int, log10(lo))
    while decade <= ceil(Int, log10(hi)) && length(ticks) < 8
        value = 10.0^decade
        if lo * 0.8 <= value <= hi * 1.2
            push!(ticks, value)
        end
        decade += 1
    end
    return isempty(ticks) ? [lo, hi] : ticks
end

function _tick_label(value::Float64, log_scale::Bool)
    if log_scale || value >= 1000 || (0 < abs(value) < 0.01)
        return @sprintf("%.0e", value)
    elseif abs(value - round(value)) < 1e-8
        return string(round(Int, value))
    else
        return @sprintf("%.3g", value)
    end
end

function _cost_svg(comparison::Comparison, subtitle::AbstractString)
    series = (
        (label = "Dead reckoning", color = "#a93226", history = comparison.dead_reckoning.cost_history),
        (label = "Spectral", color = "#1a5276", history = comparison.spectral.cost_history),
    )
    width = 900
    height = 560
    left = 110.0
    plot_right = 860.0
    top = 100.0
    bottom = 490.0
    xs = Float64[]
    ys = Float64[]
    for item in series
        for (k, value) in enumerate(item.history)
            push!(xs, Float64(k - 1))
            push!(ys, value)
        end
    end
    # A noiseless solution has cost ~1e-28. Plot that as zero, not as a 1e-27 axis.
    if maximum(ys) < 1e-8
        ys .= 0
    end
    positive = filter(>(0), ys)
    use_log = !isempty(positive) && maximum(ys) > 0 && maximum(ys) / minimum(positive) > 80
    ymap = use_log ? (y -> log10(max(y, minimum(positive) * 0.5))) : identity
    ymin, ymax = extrema(ymap.(ys))
    flat_zero = ymin == 0 && ymax == 0
    if ymin == ymax && !flat_zero
        ymin -= 1
        ymax += 1
    elseif flat_zero
        ymax = 1.0
    end
    if !flat_zero
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad
    end
    x_lo = minimum(xs)
    x_hi = maximum(xs) == x_lo ? x_lo + 1 : maximum(xs)
    xscale = (plot_right - left) / (x_hi - x_lo)
    yscale = (bottom - top) / (ymax - ymin)
    sx = x -> left + (x - x_lo) * xscale
    sy = y -> bottom - (ymap(y) - ymin) * yscale

    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
    println(io, """<rect width="100%" height="100%" fill="#ffffff"/>""")
    println(io, """<text x="40" y="40" font-family="Segoe UI, Helvetica, sans-serif" font-size="24" font-weight="600" fill="#1c1c1c">Cost versus iteration</text>""")
    println(io, """<text x="40" y="68" font-family="Segoe UI, Helvetica, sans-serif" font-size="16" fill="#555555">$(_xml(subtitle))</text>""")
    @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#222\" stroke-width=\"1.2\"/>\n", sx(x_lo), bottom, sx(x_hi), bottom)
    @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#222\" stroke-width=\"1.2\"/>\n", sx(x_lo), bottom, sx(x_lo), top)
    x_ticks = _axis_ticks(x_lo, x_hi; integer = true)
    for tick in x_ticks
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#222\" stroke-width=\"1\"/>\n", sx(tick), bottom, sx(tick), bottom + 6)
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\" font-family=\"Segoe UI, Helvetica, sans-serif\" font-size=\"13\" fill=\"#333\">%s</text>\n",
            sx(tick), bottom + 22, _tick_label(tick, false))
    end
    y_ticks = use_log ? _log_ticks(minimum(positive), maximum(ys)) : _axis_ticks(minimum(ys), maximum(ys))
    for tick in y_ticks
        y_screen = sy(tick)
        (y_screen < top - 2 || y_screen > bottom + 2) && continue
        @printf(io, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#222\" stroke-width=\"1\"/>\n", sx(x_lo) - 6, y_screen, sx(x_lo), y_screen)
        @printf(io, "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"end\" font-family=\"Segoe UI, Helvetica, sans-serif\" font-size=\"13\" fill=\"#333\">%s</text>\n",
            sx(x_lo) - 10, y_screen + 4, _tick_label(tick, use_log))
    end
    println(io, """<text x="470" y="540" text-anchor="middle" font-family="Segoe UI, Helvetica, sans-serif" font-size="16" fill="#222">Levenberg–Marquardt iteration</text>""")
    ylabel = use_log ? "cost (log scale)" : "cost"
    println(io, """<text x="22" y="300" text-anchor="middle" font-family="Segoe UI, Helvetica, sans-serif" font-size="16" fill="#222" transform="rotate(-90 22 300)">$ylabel</text>""")
    for item in series
        points = IOBuffer()
        displayed = maximum(item.history) < 1e-8 ? zeros(length(item.history)) : item.history
        for (k, value) in enumerate(displayed)
            k > 1 && print(points, " ")
            @printf(points, "%.2f,%.2f", sx(k - 1), sy(value))
        end
        @printf(io, "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"2.5\" points=\"%s\"/>\n", item.color, String(take!(points)))
        if length(displayed) <= 12
            for (k, value) in enumerate(displayed)
                @printf(io, "<circle cx=\"%.2f\" cy=\"%.2f\" r=\"3.5\" fill=\"%s\"/>\n", sx(k - 1), sy(value), item.color)
            end
        end
    end
    println(io, """<line x1="640" y1="108" x2="682" y2="108" stroke="#a93226" stroke-width="2.5"/>""")
    println(io, """<text x="690" y="113" font-family="Segoe UI, Helvetica, sans-serif" font-size="14" fill="#222">Dead reckoning</text>""")
    println(io, """<line x1="640" y1="132" x2="682" y2="132" stroke="#1a5276" stroke-width="2.5"/>""")
    println(io, """<text x="690" y="137" font-family="Segoe UI, Helvetica, sans-serif" font-size="14" fill="#222">Spectral</text>""")
    println(io, "</svg>")
    return String(take!(io))
end
