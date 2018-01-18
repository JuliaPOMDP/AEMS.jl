mutable struct ActionNode
    r::Float64      # reward for taking this action
    pind::Int       # index of parent belief node in Graph.belief_nodes
    ai::Int         # which action does this correspond to

    pab::Float64        # P(a | b)
    L::Float64
    U::Float64

    children::UnitRange{Int64}

    function ActionNode(r, pind, ai, L, U, children)
        new(r, pind, ai, 0.0, L, U, children)
    end
end


mutable struct BeliefNode{B}
    b::B            # belief associated with this node
    ind::Int        # index of this node in Graph.belief_nodes
    pind::Int       # index of parent action node in Graph.action_nodes
    aind::Int       # index of best child action node in action_nodes

    oi::Int         # index of observation corresponding with this belief
    po::Float64     # probability of seeing that observation
    poc::Float64    # prob of all observations leading here (cumulative)

    L::Float64
    U::Float64
    d::Int          # depth
    gd::Float64     # discount^depth, stored to improve computation

    children::UnitRange{Int64}
end

function BeliefNode(b, L::Float64, U::Float64)
    BeliefNode(b, 1, 0, 0, 0, 1.0, 1.0, L, U, 0, 1.0, 0:0)
end
function BeliefNode(b,ind::Int,pind::Int, oi::Int,po::Float64,poc::Float64,L::Float64,U::Float64,d::Int,gd::Float64)
    return BeliefNode(b, ind, pind, 0, oi, po, poc, L, U, d, gd, 0:0)
end

mutable struct Graph{B <: BeliefNode}
    action_nodes::Vector{ActionNode}
    belief_nodes::Vector{B}

    na::Int     # number of action nodes
    nb::Int     # number of belief nodes

    root_ind::Int
    an_root::Int

    df::Float64     # discount factor

    # constructor
    function Graph{B}(df::Real) where B
        new(ActionNode[], BeliefNode{B}[], 0, 0, 1, 1, df)
    end
end

function clear_graph!(G::Graph)
    G.action_nodes = ActionNode[]
    G.belief_nodes = BeliefNode[]
    G.na = 0
    G.nb = 0
    G.root_ind = 1
    G.an_root = 1
end

isroot(G::Graph, bn::BeliefNode) = G.root_ind == bn.ind

get_root(G::Graph) = G.belief_nodes[G.root_ind]

# assumes all beliefs can have children
# even if it is a terminal belief, I suppose action could do nothing
isfringe(bn::BeliefNode) = iszero(bn.children)

function add_node(G::Graph, an::ActionNode)
    push!(G.action_nodes, an)
    G.na += 1
end
function add_node(G::Graph, bn::BeliefNode)
    push!(G.belief_nodes, bn)   # add new node to list of belief nodes
    G.nb += 1                   # length of G.belief_nodes increases by 1
end

parent_node(G::Graph, bn::BeliefNode) = G.action_nodes[bn.pind]
parent_node(G::Graph, an::ActionNode) = G.belief_nodes[an.pind]
