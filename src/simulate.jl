######################################################################
# simulate.jl
# 
# User can update the planner with the action and observation.
# This allows planner to keep its old tree.
#
# But, original simulations have no provision for updating planner.
# This code adds modifications
######################################################################

function simulate(sim::RolloutSimulator, pomdp::POMDP{S, A, O}, policy::AEMSPlanner, updater::Updater, initial_distribution::Any) where {S, A, O}

    s = rand(sim.rng, initial_distribution)
    b = initialize_belief(updater, initial_distribution)

    disc = discount(pomdp)
    r_total = 0.0

    step = 1

    while !isterminal(pomdp, s) && step <= sim.max_steps # TODO also check for terminal observation
        a = action(policy, b)

        (sp, o, r) = @gen(:sp, :o, :r)(pomdp, s, a, sim.rng)

        r_total += disc*r

        b = update(updater, b, a, o)

        # Only change required AEMS.jl
        if policy.root_manager == :user
            update_root(policy, a, o)
        end

        disc *= discount(pomdp)
        s = sp
        step += 1
    end

    return r_total
end
