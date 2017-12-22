using D3Trees

function visualize(G::Graph)
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
            push!(text, "b0")
        else
            push!(text, "P(o$(bn.oi)) = $(round(bn.po,3))")
        end
    end

    for an in G.action_nodes
        push!(children, collect(an.children))
        push!(text, "a$(an.ai)")
    end

    D3Tree(children, text=text)
end
