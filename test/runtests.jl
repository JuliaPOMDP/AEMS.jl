using AEMS
using POMDPs
using POMDPToolbox
using POMDPModels
using Base.Test

#@requirements_info solver pomdp

######################################################################
# BABY POMDP Tests
######################################################################
pomdp = BabyPOMDP()
up = DiscreteUpdater(pomdp)

# testing solver with :clear root_manager
solver = AEMSSolver(n_iterations = 1000, max_time = .1, updater = up)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(n_iterations=1000, max_time=.1, updater=up, rm=:belief)
test_solver(solver, pomdp)

# testing solver with :user root_manager
solver = AEMSSolver(n_iterations=1000, max_time=.1, updater=up, rm=:user)
test_solver(solver, pomdp)



######################################################################
# TIGER PROBLEM TESTS
######################################################################
pomdp = TigerPOMDP()
up = DiscreteUpdater(pomdp)

# testing solver with :clear root_manager
solver = AEMSSolver(n_iterations = 1000, max_time=.1, updater=up)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(n_iterations = 1000, max_time=.1,updater=up,rm=:belief)
test_solver(solver, pomdp)

# testing solver with :belief root_manager
solver = AEMSSolver(n_iterations = 1000, max_time=.1, updater=up, rm=:user)
test_solver(solver, pomdp)
