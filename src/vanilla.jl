#######
# DEFINITELY
#######
# TODO: fix lower_bound and upper_bound functions
# TODO: return the action that should be taken

#######
# MAYBE
#######
# TODO: store gamma ^ d?


struct DefaultPolicy <: Policy end

# SOLVER
mutable struct AEMSSolver{U<:Updater, PL<:Policy, PU<:Policy} <: Solver
    n_iterations::Int
    max_time::Float64   # max time per action, in seconds
    updater::U

    lb::PL
    ub::PU
end
function AEMSSolver(;n_iterations::Int=100, max_time::Float64=1.0, updater=DiscreteUpdater(p), lb=DefaultPolicy(), ub=DefaultPolicy())
    AEMSSolver(n_iterations, max_time, updater, lb, ub)
end


# PLANNER
mutable struct AEMSPlanner{P<:POMDP, U<:Updater, PL<:Policy, PU<:Policy} <: Policy
    solver::AEMSSolver  # contains solver parameters
    pomdp::P            # model
    updater::U
    G::Graph
    lb::PL      # lower bound
    ub::PU      # upper bound
end
function AEMSPlanner(s::AEMSSolver, p::POMDP, lb, ub)
    return AEMSPlanner(s, p, s.updater, Graph(discount(p)), lb, ub)
end


# SOLVE
function solve(solver::AEMSSolver, pomdp::POMDP)

    # if no lower bound was given to solver, default to blind
    lb = solver.lb
    if typeof(lb) == DefaultPolicy
        lb = BlindPolicy(pomdp)
    end

    # if no upper bound was passed to solver, default to FIB
    ub = solver.ub
    if typeof(ub) == DefaultPolicy
        ub = solve(FIBSolver(), pomdp)
    end

    AEMSPlanner(solver, pomdp, lb, ub)
end

updater(planner::AEMSPlanner) = planner.updater


# TODO: don't recreate the graph
function action(policy::AEMSPlanner, b)

    t_start = time()

    # recreate the graph
    clear_graph!(policy.G)     # TODO: don't do this shit

    # create belief node and put it in graph
    L = value(policy.lb, b)
    U = value(policy.ub, b)
    bn_root = BeliefNode(b, 1, 0, 0, 1.0, L, U, 0)
    add_node(policy.G, bn_root)

    for i = 1:policy.solver.n_iterations
        
        # determine node to expand
        best_bn = select_node(policy.G)
        Lold, Uold = best_bn.L, best_bn.U

        expand(policy, best_bn)

        backtrack(policy.G, best_bn, Lold, Uold)

        # stop if the timeout has been reached
        if (time() - t_start) > policy.solver.max_time
            #println("max time reached")
            break
        end
    end

    # now return the best action
    best_L = -Inf
    best_ai = 1
    for ci in bn_root.children
        an = policy.G.action_nodes[ci]
        if an.L >= best_L
            best_L = an.L
            best_ai = an.ai
        end
    end

    #a = actions(policy.pomdp)[best_ai]
    #println("a = ", a)
    return actions(policy.pomdp)[best_ai]
end

function backtrack(G::Graph, bn::BeliefNode, Lold::Float64, Uold::Float64)

    while !isroot(G, bn)
        an = parent_node(G, bn)
        an.L += G.df*bn.po * (bn.L - Lold)
        an.U += G.df*bn.po * (bn.U - Uold)
        bn = parent_node(G, an)

        # loop over children and set P(a | b)
        # TODO: must I do this even if new bn's bounds aren't improved?
        #  maybe, if action bounds change (AEMS1)
        update_pab(G, bn)

        # if belief bounds are improved
        if an.L > bn.L || an.U < bn.U
            Lold = bn.L
            Uold = bn.U

            bn.L = an.L
            bn.U = an.U
        else
            break   # if bounds not improved
        end
    end
end

# updates P(a | b) for all action children of node bn
#  according to AEMS2 heuristic
function update_pab(G::Graph, bn::BeliefNode)
    best_ai = bn.children[1]
    best_U = -Inf
    for ai in bn.children
        G.action_nodes[ai].pab = 0.0
        if G.action_nodes[ai].U > best_U
            best_U = G.action_nodes[ai].U
            best_ai = ai
        end
    end
    G.action_nodes[best_ai].pab = 1.0
end

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

# evaluates a fringe node using the AEMS heuristic
function evaluate_node(G::Graph, bn::BeliefNode)

    # compute pb = P(b^d)
    pb = 1.0
    cn = bn     # current node cn
    while !isroot(G, cn)
        an = parent_node(G, cn)
        cn = parent_node(G, an)

        pb *= an.pab * cn.po
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

    # for ease of notation
    G = p.G
    b = bn.b
    pomdp = p.pomdp
    action_list = actions(pomdp)
    obs_list = observations(pomdp)

    # first remove belief node from fringe list
    remove_from_fringe(G, bn)

    bestL = bestU = -Inf
    a_start = G.na + 1
    for (ai,a) in enumerate(action_list)
        aind = G.na + 1 # index of parent action node
        b_start = G.nb + 1

        # reward for action node
        La = Ua = r = R(pomdp, b, a)

        for (oi,o) in enumerate(obs_list)
            # probability of measuring o
            po = O(pomdp, b, a, o)

            # update belief
            bp = update(p.updater, b, a, o)

            # determine bounds at new belief
            L = value(p.lb, bp)
            U = value(p.ub, bp)

            La += G.df * po * L
            Ua += G.df * po * U

            # create belief node and add to graph
            bpn = BeliefNode(bp, G.nb+1, aind, oi, po, L, U, bn.d+1)
            add_node(G, bpn)
        end

        # create action node and add to graph
        (Ua > bestU) && (bestU = Ua)
        (La > bestL) && (bestL = La)

        # range for action node
        b_range = b_start:G.nb

        an = ActionNode(r, bn.ind, b_range, ai, La, Ua)
        add_node(G, an)
    end

    bn.children = a_start:G.na

    # TODO: check that this is right. I think so
    #  don't need to do max(bn.L, bestL)
    #   before it was just an approximation. now it's slightly better
    bn.L = bestL
    bn.U = bestU

    # now iterate over actions to compute P(a | b)
    update_pab(G, bn)
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
        expected_r += reward(pomdp, s, a) * pdf(b, s)
    end
    return expected_r
end
