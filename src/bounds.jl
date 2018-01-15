# handles upper and lower bounds
# Point-Based POMDP Algorithms: Improved Analysis and Implementation
#  search for "blind"

export BlindPolicy, FixedActionSolver

mutable struct BlindPolicy{A} <: Policy
    a::A
    v::Float64
end

function BlindPolicy(pomdp::POMDP)
    action_list = actions(pomdp)
    state_list = ordered_states(pomdp)  # TODO: could it just be states?

    best_action = action_list[1]
    best_worst_reward = -Inf

    for a in action_list
        worst_reward = Inf
        for s in state_list
            r_sa = reward(pomdp, s, a)
            worst_reward = min(r_sa, worst_reward)
        end

        if worst_reward > best_worst_reward
            best_worst_reward = worst_reward
            best_action = a
        end
    end

    v = best_worst_reward / (1.0 - discount(pomdp))

    return BlindPolicy(best_action, v)
end

action(policy::BlindPolicy, b) = policy.a

value(policy::BlindPolicy, b) = policy.v

struct FixedActionSolver <: Solver
    max_iterations::Int64
    tolerance::Float64
end
function FixedActionSolver(;max_iterations::Int=100,tolerance::Float64=1e-3)
    return FixedActionSolver(max_iterations, tolerance)
end

function solve(solver::FixedActionSolver, pomdp::POMDP)

    # convenience variables
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    state_list = ordered_states(pomdp)
    action_list = ordered_actions(pomdp)

    # fill with max_worst_value
    bp = BlindPolicy(pomdp)
    max_worst_value = bp.v
    alphas = ones(ns,na) * max_worst_value
    old_alphas = ones(ns,na) * max_worst_value
    

    for i = 1:solver.max_iterations
        copy!(old_alphas, alphas)
        residual = 0.0

        for (ai,a) in enumerate(action_list)
            for (si,s) in enumerate(state_list)
                sp_dist = transition(pomdp, s, a)
                sp_sum = 0.0
                for (spi, sp) in enumerate(state_list)
                    sp_sum += pdf(sp_dist, sp) * old_alphas[spi,ai]
                end
                r = reward(pomdp, s, a)
                alphas[si,ai] = r + discount(pomdp) * sp_sum

                alpha_diff = abs(alphas[si,ai] - old_alphas[si,ai])
                residual = max(alpha_diff, residual)
            end
        end

        residual < solver.tolerance ? break : nothing
    end

    return AlphaVectorPolicy(pomdp, alphas)
end
