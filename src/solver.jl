######################################################################
# solver.jl
#
# defines solver and planner types and provides convenience functions
#
# action(planner, b) function provided in action.jl
######################################################################

struct DefaultPolicy <: Policy end
struct DefaultUpdater <: Updater end

"""
Implements anytime error minimization search (AEMS).
Specifically, this is AEMS2, which generally outperforms AEMS1.

Example with some optional arguments:

`AEMSSolver(max_iterations = 100, action_selector = :L)`

Fields:

    
- `max_iterations`
    Maximum node expansions per action.

- `max_time`
    Maximum time (in seconds) to spend per action.

- `updater`
    Updater used to propagate belief nodes. Defaults to DiscreteUpdater.

- `lower_bound`
    Subtype of `Policy`. Defaults to fixed action policy.

- `upper_bound`
    Subtype of `Policy`. Defaults to FIB policy.

- `root_manager`
    Allowable values are `:clear`, `:belief`, or `:user`.
    Defaults to `:clear`.

- `action_selector`
    Allowable values are `:U` or `:L`. Defaults to `:U`
"""
mutable struct AEMSSolver{U<:Updater, PL<:Policy, PU<:Policy} <: Solver
    max_iterations::Int
    max_time::Float64   # max time per action, in seconds

    updater::U

    lower_bound::PL
    upper_bound::PU

    root_manager::Symbol
    action_selector::Symbol
end
function AEMSSolver(;
                    max_iterations::Int = 1000,
                    max_time::Real = 1.0,
                    updater = DefaultUpdater(),
                    lower_bound = DefaultPolicy(),
                    upper_bound = DefaultPolicy(),
                    root_manager::Symbol = :clear,
                    action_selector::Symbol = :U
                   )

    @assert in(action_selector, (:U, :L))
    @assert in(root_manager, (:clear, :belief, :user))

    return AEMSSolver(max_iterations, float(max_time), updater, lower_bound, upper_bound, root_manager, action_selector)
end


# planner
struct AEMSPlanner{S <: AEMSSolver,
                   P <: POMDP,
                   U <: Updater,
                   PL <: Policy,
                   PU <: Policy,
                   GT <: Graph,
                   A,
                   O
                  } <: Policy

    solver::S           # contains solver parameters
    pomdp::P            # model
    updater::U
    G::GT
    lower_bound::PL      # lower bound
    upper_bound::PU      # upper bound
    action_list::Vector{A}
    obs_list::Vector{O}
    root_manager::Symbol
end
function AEMSPlanner(s, pomdp::POMDP, up, lb, ub)
    b0 = initialize_belief(up, initialstate(pomdp))
    bn_type = BeliefNode{typeof(b0)}
    G = Graph{bn_type}(discount(pomdp))
    a_list = ordered_actions(pomdp)
    o_list = ordered_observations(pomdp)
    rm = s.root_manager
    return AEMSPlanner(s, pomdp, up, G, lb, ub, a_list, o_list, rm)
end

"""
Clears existing action/observation graph.
"""
clear_graph!(planner::AEMSPlanner) = clear_graph!(planner.G)

"""
Returns tuple `(nb, na)`, numbers of belief and action nodes in graph
"""
count_nodes(planner::AEMSPlanner) = planner.G.nb, planner.G.na

function update_root(planner::AEMSPlanner, a, o)
    if planner.root_manager != :user
        error("User is trying to update root but (root_manager != :user).")
    end
    ai = actionindex(planner.pomdp, a)
    oi = obsindex(planner.pomdp, o)

    original_root = get_root(planner.G)

    an_ind = original_root.children.start + ai - 1
    an = get_an(planner, an_ind)

    new_root_ind = an.children.start + oi - 1
    planner.G.root_ind = new_root_ind
    return planner      # to prevent it from returning planner.G.root_ind
end

# convenience functions
get_bn(planner::AEMSPlanner, bn_idx::Int) = planner.G.belief_nodes[bn_idx]
get_an(planner::AEMSPlanner, an_idx::Int) = planner.G.action_nodes[an_idx]


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

function determine_root_node(planner::AEMSPlanner, b)
    if planner.root_manager == :clear
        clear_graph!(planner.G)
    end
    if planner.G.nb == 0
        L = value(planner.lower_bound, b)
        U = value(planner.upper_bound, b)
        bn_root = BeliefNode(b, L, U)
        add_node(planner.G, bn_root)
        return bn_root
    end

    if planner.root_manager == :belief
        bn_root = get_root(planner.G)

        # if this root has same belief, make it the root
        if bn_root.b == b
            return bn_root
        end

        # search through children to see if they might be root
        for an_idx in bn_root.children
            an = get_an(planner, an_idx)
            for bn_idx in an.children
                bn = get_bn(planner, bn_idx)
                if bn.b == b
                    planner.G.root_ind = bn.ind
                    return get_root(planner.G)
                end
            end
        end
        error("None of the child beliefs match input")
        # another option is to clear graph and start again
    end

    # if we get to here, then we must have :user
    return get_root(planner.G)
end
