# TODO: store gamma ^ d
# TODO: add time cutoff


mutable struct ActionNode
    r::Float64
    parent_ind::Int
    children::UnitRange{Int64}
    ai::Int     # which action does this correspond to

    pab::Float64        # P(a | b)
    U::Float64
    L::Float64

    ActionNode(r, pind, children, ai) = new(r, pind, children, ai, 1.0, 0.0,0.0)
end


mutable struct BeliefNode{B}
    b::B
    ind::Int
    pind::Int       # index of parent node

    oi::Int         # index of observation corresponding with this belief
    po::Float64     # probability of seeing that observation

    L::Float64
    U::Float64
    d::Int          # depth

    children::UnitRange{Int64}
end

function BeliefNode(b,ind::Int,pind::Int,oi::Int,po::Float64,L::Float64,U::Float64,d::Int)
    return BeliefNode(b, ind, pind, oi, po, L, U, d, 0:0)
end

mutable struct Graph
    action_nodes::Vector{ActionNode}
    belief_nodes::Vector{BeliefNode}

    na::Int     # number of action nodes
    nb::Int     # number of belief nodes

    root_ind::Int
    fringe_list::Set{BeliefNode}

    df::Float64     # discount factor

    # constructor
    function Graph(df::Real)
        new(ActionNode[], BeliefNode[], 0, 0, 1, Set{BeliefNode}(), df)
    end
end

function add_node(G::Graph, an::ActionNode)
    push!(G.action_nodes, an)
    G.na += 1
end

function add_node(G::Graph, bn::BeliefNode)
    push!(G.belief_nodes, bn)   # add new node to list of belief nodes
    push!(G.fringe_list, bn)    # add new node to fringe list
    G.nb += 1                   # length of G.belief_nodes increases by 1
end


mutable struct AEMSSolver
    n_iterations::Int

    AEMSSolver(n_iterations::Int=100) = new(n_iterations)
end

mutable struct AEMSPlanner{P<:POMDP, U<:Updater}
    solver::AEMSSolver  # contains solver parameters
    pomdp::P            # model
    updater::U
    G::Graph

    #AEMSPlanner(s, p, u, G) = new(s, p, u, G)
end


# TODO: is this outer constructor ok?
function AEMSPlanner(s::AEMSSolver, p::POMDP)
    return AEMSPlanner(s, p, DiscreteUpdater(p), Graph(discount(p)))
end

function AEMSPlanner(s::AEMSSolver, p::POMDP, u::Updater)
    return AEMSPlanner(s, p, u, Graph(discount(p)))
end

function solve(solver::AEMSSolver, pomdp::POMDP, up::Updater)
    AEMSPlanner(solver, pomdp, up)
end


# TODO: don't recreate the graph
function action(policy::AEMSPlanner, b)

    # create belief node and put it in graph
    L = 10.0
    U = 0.0
    pind = 0
    bn_root = BeliefNode(b, 1, pind, 0, 0.0, L, U, 0, 0:0)
    add_node(policy.G, bn_root)

    # TODO: make a different cutoff criterion
    for i = 1:policy.solver.n_iterations

        # determine best node
        best_bn = select_node(policy.G)

        expand(policy, best_bn)

        #backtrack(policy.pomdp, b)
        cn = best_bn
        while cn.ind != 1
            old_L = cn.L
            old_U = cn.U

            an = policy.G.action_nodes[cn.pind]

            #U = an.U - old_U * discount


            # now update L and U for
            cn = policy.G.belief_nodes[an.parent_ind]

            # now iterate over all child nodes to do
        end
    end
    
    return policy.G
end

# TODO: evaluate nodes properly
# return best belief node
function select_node(G::Graph)
    best_bn = G.belief_nodes[1]
    best_val = -Inf

    # iterate over fringe list, evaluating each node
    for bn in G.fringe_list
        bn_val = evaluate_node(G, bn)
        if bn_val >= best_val
            best_bn = bn
            best_val = bn_val
        end
    end

    return best_bn
end

# evaluates a fringe node
function evaluate_node(G::Graph, bn::BeliefNode)

    pb = 1.0    # P(b^d)
    cn = bn     # current node cn
    while cn.ind != 1
        an = G.action_nodes[cn.pind]
        cn = G.belief_nodes[an.parent_ind]

        pb = cn.po * an.pab
    end

    return G.df^bn.d * pb * (bn.U - bn.L)
end



# one-step look-ahead, from fringe belief state given in parameter
# constructs action AND-nodes and belief state OR-nodes
#   resulting from all possible action and observations
# Also computes all lower and upper bounds for next belief states
#
# I'll also have it do one step of backtracking
function expand(p::AEMSPlanner, bn::BeliefNode)

    # first remove belief node from fringe list
    delete!(p.G.fringe_list, bn)

    # consider all actions we can take
    pomdp = p.pomdp
    action_list = actions(pomdp)
    na = n_actions(p.pomdp)
    obs_list = observations(pomdp)

    # for ease of notation
    G = p.G
    b = bn.b

    a_start = G.na + 1
    for (ai,a) in enumerate(action_list)
        aind = G.na + 1 # index of parent action node
        b_start = G.nb + 1

        for (oi,o) in enumerate(obs_list)
            # probability of measuring o
            po = O(pomdp, b, a, o)

            # update belief
            bp = update(p.updater, b, a, o)

            # determine bounds
            # TODO: need way to assign upper and lower bounds
            L = 0.0
            U = 10.0

            # create belief node and add to graph
            bpn = BeliefNode(bp, G.nb+1, aind, oi, po, L, U, bn.d+1)
            add_node(G, bpn)
        end

        # create action node and add to graph
        r = R(pomdp, b, a)
        b_range = b_start:G.nb
        an = ActionNode(r, bn.ind, b_range, ai)
        add_node(G, an)
    end

    bn.children = a_start:G.na
end

function O(pomdp, b, a, o)
    state_list = ordered_states(pomdp)
    sum_sp = 0.0
    for (spi,sp) in enumerate(state_list)
        od = observation(pomdp, a, sp)
        po = pdf(od, o)
        sum_s = 0.0
        for (si,s) in enumerate(state_list)
            spd = transition(pomdp, s, a)
            sum_s += pdf(spd, sp) * pdf(b, s)
        end
        sum_sp += sum_s * po
    end
    return sum_sp
end

# TODO: maybe don't assume that we have access to ordered_states?
function R(pomdp, b, a)
    state_list = ordered_states(pomdp)
    expected_r = 0.0
    for s in state_list
        expected_r += reward(pomdp, s, a)# * pdf(b, s)
    end
    return expected_r
end

function backup(bn::BeliefNode)
    for ci in an.children

    end
    # return the parent belief
end
