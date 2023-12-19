using AEMS
using POMDPs
using POMDPTools
using POMDPModels
using Test

#@requirements_info solver pomdp

######################################################################
# BABY POMDP Tests
######################################################################
pomdp = BabyPOMDP()
up = DiscreteUpdater(pomdp)

# testing solver with :clear root_manager
solver = AEMSSolver(max_time=0.1, updater=up)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(max_time=0.1, updater=up, root_manager=:belief)
test_solver(solver, pomdp)

# testing solver with :user root_manager
solver = AEMSSolver(max_time=0.1, updater=up, root_manager=:user)
test_solver(solver, pomdp)

# Test simulate 
solver = AEMSSolver(max_time=0.1, updater=up, root_manager=:user)
policy = solve(solver, pomdp)
try
    r = simulate(RolloutSimulator(max_steps=15), BabyPOMDP(), policy, solver.updater, initialstate(pomdp))
    @test r != 0.0
catch
    @test false
end


######################################################################
# TIGER PROBLEM TESTS
######################################################################
pomdp = TigerPOMDP()
up = DiscreteUpdater(pomdp)

# testing solver with :clear root_manager
solver = AEMSSolver(max_time=0.1, updater=up)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(max_time=0.1, updater=up, root_manager=:belief)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(max_time=0.1, updater=up, root_manager=:user)
test_solver(solver, pomdp)
