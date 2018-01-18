# AEMS

[![Build Status](https://travis-ci.org/JuliaPOMDP/AEMS.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/AEMS.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/AEMS.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/AEMS.jl?branch=master)

Implements anytime error minimization search (AEMS) solver for POMDPs. This algorithm was originally described in

> Ross, Stéphane, and Brahim Chaib-Draa. "AEMS: An Anytime Online Search Algorithm for Approximate Policy Refinement in Large POMDPs." IJCAI. 2007.

This solver uses AEMS2, which outperforms AEMS1 in nearly all published experiments.

# Installation

```julia
Pkg.clone("https://github.com/JuliaPOMDP/AEMS.jl")
```


# Quick Use

```julia
using POMDPs, POMDPToolbox, AEMS, POMDPModels

pomdp = BabyPOMDP()
solver = AEMSSolver()
planner = solve(solver, pomdp)
```

# Solver Options

* `n_iterations` Maximum number of fringe expansions during one action. Defaults to 1000.
* `max_time` Maximum time (in seconds) to spend on one action. Defaults to 1 second.
* `updater` The updater used to propagate beliefs in the tree. Defaults to a discrete updater.
* `lower_bound` Defaults to a fixed-action policy.
* `upper_bound` Defaults to a policy generated by FIB.
* `root_manager` Determines how the root changes once an action is taken and an observation is received. Allowed values are `:clear`, `:belief`, `:user`. Defaults to `:clear`.

# Visualization
Once you have a planner and have called `action`, you can use the following code to bring up an interactive tree in a Chrome browser window. Click a node to expand/unexpand it.

```julia
using D3Trees

tree = D3Tree(planner)      # creates a visualization tree
inchrome(tree)              # opens chrome tab to show visualization tree
```
