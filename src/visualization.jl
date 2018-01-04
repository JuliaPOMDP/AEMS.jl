using D3Trees

function visualize(planner::AEMSPlanner)
    G = planner.G
    children = Array{Vector{Int}}(0)
    text = String[]

    # create vector of vectors
    for (bni,bn) in enumerate(G.belief_nodes)
        if bn.children != 0:0
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
        pab = "\nP(a|b)=$(round(an.pab,2))"
        push!(text, string(r,U,L,pab))
    end

    D3Tree(children, text=text)
end
