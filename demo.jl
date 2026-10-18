# Tuesday demo for the 2D pose-graph project.
#
# From this folder:
#   julia --project=. demo.jl
#   julia --project=. -e "using Pkg; Pkg.test()"

using PoseGraphSLAM
using Printf

const FIGURES = joinpath(@__DIR__, "figures")

function _deg(radians)
    return radians * 180 / π
end

function _report(io::IO, name::AbstractString, graph::PoseGraph, comparison::Comparison)
    println(io, name)
    println(io, "-"^length(name))
    @printf(io, "%-16s %12s %12s %10s %12s %14s %12s\n",
        "start", "init cost", "final cost", "iters", "pos RMSE", "heading RMSE", "gradient")
    for result in (comparison.dead_reckoning, comparison.spectral)
        err = pose_error(result.poses, graph.truth)
        @printf(io, "%-16s %12.4g %12.4g %10d %12.4g %10.3f deg %12.3g\n",
            init_label(result.init),
            result.init_cost,
            result.cost,
            result.iterations,
            err.pos_rmse,
            _deg(err.heading_rmse),
            result.gradient_norm)
    end
    truth = graph.truth
    dr0 = comparison.dead_reckoning.init_poses
    dθ = wrap_angle(dr0[end].θ - truth[end].θ)
    @printf(io, "Dead-reckoning heading error at the last pose, before refinement: %.1f deg\n", _deg(dθ))
    dr = comparison.dead_reckoning
    sp = comparison.spectral
    dr_err = pose_error(dr.poses, truth)
    sp_err = pose_error(sp.poses, truth)
    if sp_err.pos_rmse < 0.5 && dr_err.pos_rmse > max(1.0, 5 * sp_err.pos_rmse) && dr.cost > 2 * sp.cost
        println(io, "Dead reckoning stopped in a worse basin. The spectral start finished near the ground truth.")
    elseif dr_err.pos_rmse < 0.5 && sp_err.pos_rmse < 0.5
        println(io, "Both starts finished near the ground truth.")
    end
    println(io)
    return nothing
end

function main()
    mkpath(FIGURES)
    report_path = joinpath(FIGURES, "results.txt")
    open(report_path, "w") do io
        toy = toy_graph()
        gap = jacobian_discrepancy(toy, toy_evaluation_state(toy))
        line = @sprintf("Toy graph, 3 poses and 3 edges. Max |analytic Jacobian − finite difference| = %.3e\n", gap)
        print(io, line)
        print(line)
        println(io)

        scenarios = (
            ("Noiseless loop", "noiseless", noiseless_graph(; n = 16, radius = 8.0), 30),
            ("Moderate noise, one loop closure", "moderate", moderate_graph(), 40),
            ("Grid, large heading noise", "hard_grid", hard_graph(), 80),
        )
        for (title, filename, graph, iters) in scenarios
            comparison = compare(graph; max_iters = iters, grad_tol = 1e-6)
            _report(io, title, graph, comparison)
            _report(stdout, title, graph, comparison)
            save_figures(FIGURES, filename, graph, comparison; subtitle = title)
        end

        println(io, "Disconnected graph")
        println(io, "------------------")
        try
            solve(disconnected_graph())
            println(io, "ERROR: a solution was returned for a disconnected graph")
        catch err
            if err isa DisconnectedGraphError
                println(io, sprint(showerror, err))
            else
                rethrow()
            end
        end
        println(io)
    end
    println("Wrote figures and ", report_path)
    return nothing
end

main()
