
struct DefaultPolicy <: Policy end
struct DefaultUpdater <: Updater end

# SOLVER
mutable struct AEMSSolver{U<:Updater, PL<:Policy, PU<:Policy} <: Solver
    n_iterations::Int
    max_time::Float64   # max time per action, in seconds
    updater::U

    lower_bound::PL
    upper_bound::PU
end
function AEMSSolver(;n_iterations::Int=100, max_time::Float64=1.0, updater=DefaultUpdater(), lb=DefaultPolicy(), ub=DefaultPolicy())
    AEMSSolver(n_iterations, max_time, updater, lb, ub)
end


# PLANNER
struct AEMSPlanner{S<: AEMSSolver, P<:POMDP, U<:Updater, PL<:Policy, PU<:Policy, G<:Graph} <: Policy
    solver::S           # contains solver parameters
    pomdp::P            # model
    updater::U
    G::G
    lower_bound::PL      # lower bound
    upper_bound::PU      # upper bound
end
function AEMSPlanner(s, p::POMDP, up, lb, ub)
    b0 = initialize_belief(up, initial_state_distribution(p))
    bn_type = BeliefNode{typeof(b0)}
    return AEMSPlanner(s, p, up, Graph{bn_type}(discount(p)), lb, ub)
end


# SOLVE
function solve(solver::AEMSSolver, pomdp::POMDP)

    # if no updater was given, default to discrete updater
    up = solver.updater
    if typeof(up) == DefaultUpdater
        up = DiscreteUpdater(pomdp)
    end

    # if no lower bound was given to solver, default to blind
    lb = solver.lower_bound
    if typeof(lb) == DefaultPolicy
        #lb = BlindPolicy(pomdp)
        lb = solve(FixedActionSolver(), pomdp)
    end

    # if no upper bound was passed to solver, default to FIB
    ub = solver.upper_bound
    if typeof(ub) == DefaultPolicy
        ub = solve(FIBSolver(), pomdp)
    end

    AEMSPlanner(solver, pomdp, up, lb, ub)
end

updater(planner::AEMSPlanner) = planner.updater


# TODO: don't recreate the graph
function action(policy::AEMSPlanner, b)

    t_start = time()

    # recreate the graph
    clear_graph!(policy.G)     # TODO: don't do this shit

    # create belief node and put it in graph
    L = value(policy.lower_bound, b)
    U = value(policy.upper_bound, b)
    bn_root = BeliefNode(b, L, U)
    add_node(policy.G, bn_root)

    for i = 1:policy.solver.n_iterations
        
        # determine node to expand
        best_bn = select_node(policy.G, bn_root)
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

    return actions(policy.pomdp)[best_ai]
end

# 
function backtrack(G::Graph, bn::BeliefNode, Lold::Float64, Uold::Float64)

    while !isroot(G, bn)
        an = parent_node(G, bn)
        an.L += G.df*bn.po * (bn.L - Lold)
        an.U += G.df*bn.po * (bn.U - Uold)
        bn = parent_node(G, an)

        # if belief bounds are not improved... does it matter?
        #if an.U > bn.U && an.L < bn.L .... break?

        Lold, Uold = update_node(G, bn)
    end
end

# also updates L and U given children
# updates P(a | b) for all action children of node bn
#  according to AEMS2 heuristic
function update_node(G::Graph, bn::BeliefNode)
    L_old, U_old = bn.L, bn.U
    ai_max = bn.children[1]
    U_max = L_max = -Inf
    for ai in bn.children
        an = G.action_nodes[ai]
        #an.pab = 0.0            # AEMS2
        if an.U > U_max
            U_max = an.U
            ai_max = ai        # AEMS2
        end
        an.L > L_max && (L_max = an.L)
    end

    bn.L = L_max
    bn.U = U_max

    #G.action_nodes[ai_max].pab = 1.0    # AEMS2
    bn.aind = ai_max                    # AEMS2

    return L_old, U_old
end





# recursively selects best fringe node
function select_node(G::Graph, bn::BeliefNode)
    isfringe(bn) && return bn

    # select next an
    an = G.action_nodes[bn.aind]

    # iterate over child belief nodes of an, selecting best one
    best_val = -Inf
    best_bn = bn
    for i in an.children
        bn = select_node(G, G.belief_nodes[i])
        bv = bn.poc * (bn.U - bn.L) * bn.gd
        if bv >= best_val
            best_val = bv
            best_bn = bn
        end
    end
    return best_bn
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

    La_max = Ua_max = -Inf  # max Ua over actions
    an_start = G.na + 1
    ai_max = an_start       # index of maximum upper bound (for AEMS2)

    for (ai,a) in enumerate(action_list)
        an_ind = G.na + 1 # index of parent action node
        bn_start = G.nb + 1

        # bounds and reward for action node
        La = Ua = r = R(pomdp, b, a)

        for (oi,o) in enumerate(obs_list)
            # probability of measuring o
            po = O(pomdp, b, a, o)

            # update belief
            bp = update(p.updater, b, a, o)

            # determine bounds at new belief
            L = value(p.lower_bound, bp)
            U = value(p.upper_bound, bp)

            La += G.df * po * L
            Ua += G.df * po * U

            # create belief node and add to graph
            poc = bn.poc * bn.po
            bpn = BeliefNode(bp, G.nb+1, an_ind, oi, po, poc, L, U, bn.d+1, bn.gd*G.df)
            add_node(G, bpn)
        end

        # update
        if Ua > Ua_max
            Ua_max = Ua
            ai_max = an_ind     # AEMS2
        end
        La > La_max && (La_max = La)

        # create action node and add to graph
        b_range = bn_start:G.nb      # indices of children belief nodes
        an = ActionNode(r, bn.ind, ai, La, Ua, b_range)
        add_node(G, an)
    end

    # child nodes of bn are the action nodes we've opened
    bn.children = an_start:G.na

    bn.L = La_max
    bn.U = Ua_max

    #G.action_nodes[ai_max].pab = 1.0    # AEMS2
    bn.aind = ai_max                    # AEMS2

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
