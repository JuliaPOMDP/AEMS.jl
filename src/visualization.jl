######################################################################
# visualization.jl
#
# extends D3Trees to make a graph
#
# TODO:
#  Don't enumerate over belief nodes; start from root and move down.
######################################################################

using D3Trees

function D3Trees.D3Tree(planner::AEMSPlanner; title="AEMS Tree", kwargs...)
    G = planner.G
    children = Array{Vector{Int}}(0)
    text = String[]

    # create vector of vectors
    for (bni,bn) in enumerate(G.belief_nodes)
        if !isfringe(bn)
            push!(children, collect(bn.children) + G.nb)
        else
            push!(children, [])
        end
        if bni == 1
            push!(text, "b0\nU=$(round(bn.U,2))\nL=$(round(bn.L,2))")
        else
            p = "P(o$(bn.oi)) = $(round(bn.po,2))"
            L = "\nL=$(round(bn.L,2))"
            U = "\nU=$(round(bn.U,2))"
            push!(text, string(p,U,L))
        end
    end

    for an in G.action_nodes
        push!(children, collect(an.children))
        r = "r(a$(an.ai)) = $(round(an.r))"
        L = "\nL=$(round(an.L,2))"
        U = "\nU=$(round(an.U,2))"
        push!(text, string(r,U,L))
    end

    D3Tree(children, text=text, title=title, kwargs...)
end
