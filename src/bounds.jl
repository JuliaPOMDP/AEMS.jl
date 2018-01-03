# handles upper and lower bounds
# Point-Based POMDP Algorithms: Improved Analysis and Implementation
#  search for "blind"

export BlindPolicy

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


