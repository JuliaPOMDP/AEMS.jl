######################################################################
# action.jl
# 
# defines the action function.
#
# also provides most of the helper functions called by action.
######################################################################

function action(planner::AEMSPlanner, b)
    t_start = time()

    bn_root = determine_root_node(planner, b)

    for i = 1:planner.solver.max_iterations
        
        # determine node to expand and its pre-expansion bounds
        best_bn = select_node(planner.G, bn_root)
        Lold, Uold = best_bn.L, best_bn.U

        expand(planner, best_bn)

        backtrack(planner.G, best_bn, Lold, Uold)

        # stop if the timeout has been reached
        if (time() - t_start) > planner.solver.max_time
            break
        end
    end

    # now return the best action
    return get_best_action(planner, bn_root)
end

# allows user to select action node with best upper or lower bound
function get_best_action(planner::AEMSPlanner, bn_root::BeliefNode)
    # if upper bound is desired...
    best_ai = get_an(planner, bn_root.aind).ai

    # if lower bound is desired...
    if planner.solver.action_selector == :L
        # loop over child action nodes to find best lower bound
        best_L = -Inf
        for an_idx in bn_root.children
            an = get_an(planner, an_idx)
            if an.L > best_L
                best_L = an.L
                best_ai = an.ai
            end
        end
    end
    return planner.action_list[best_ai]
end



# recursively selects best fringe node for expansion
function select_node(G::Graph, bn::BeliefNode)
    isfringe(bn) && return bn

    # select next an
    an = G.action_nodes[bn.aind]

    # iterate over child belief nodes of an, selecting best one
    best_val = -Inf
    best_bn = bn
    for bn_idx in an.children
        bn = select_node(G, G.belief_nodes[bn_idx])
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

    La_max = Ua_max = -Inf  # max Ua over actions
    an_start = G.na + 1
    ai_max = an_start       # index of maximum upper bound (for AEMS2)

    for (ai,a) in enumerate(p.action_list)
        an_ind = G.na + 1 # index of parent action node
        bn_start = G.nb + 1

        # bounds and reward for action node
        La = Ua = r = R(pomdp, b, a)

        for (oi,o) in enumerate(p.obs_list)
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
    bn.aind = ai_max        # AEMS2
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
        if an.U > U_max
            U_max = an.U
            ai_max = ai        # AEMS2
        end
        an.L > L_max && (L_max = an.L)
    end

    bn.L = L_max
    bn.U = U_max
    bn.aind = ai_max        # AEMS2

    return L_old, U_old
end



# probability of observing o after taking a from b
# TODO: use iterator... plus I'm not sure how to make that work
# TODO: I'm not sure this works
function O(pomdp::POMDP, b, a, o)
    state_list = ordered_states(pomdp)
    sum_sp = 0.0
    for sp in state_list
        od = observation(pomdp, a, sp)
        po = pdf(od, o)
        if po == 0
            continue
        end
        sum_s = 0.0
        for s in state_list
            spd = transition(pomdp, s, a)
            sum_s += pdf(spd, sp) * pdf(b, s)
        end
        sum_sp += sum_s * po
    end
    return sum_sp
end
function O(pomdp::POMDP, b::DiscreteBelief, a, o)
    sum_sp = 0.0
    for sp in b.state_list
        od = observation(pomdp, a, sp)
        po = pdf(od, o)
        if po == 0
            continue
        end
        sum_s = 0.0
        for s in b.state_list
            spd = transition(pomdp, s, a)
            sum_s += pdf(spd, sp) * pdf(b, s)
        end
        sum_sp += sum_s * po
    end
    return sum_sp
end

# expected reward 
# TODO: test thoroughly... this is pretty but maybe inefficient?
R(p::POMDP, b, a) = sum( reward(p,s,a)*pdf(b,s) for s in iterator(b) )

# This is equivalent to above, but it allocates less memory
function R(pomdp::POMDP, b::DiscreteBelief, a)
    expected_r = 0.0
    for s in b.state_list
        expected_r += reward(pomdp, s, a) * pdf(b, s)
    end
    return expected_r
end
