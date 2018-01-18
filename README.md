# AEMS

[![Build Status](https://travis-ci.org/JuliaPOMDP/AEMS.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/AEMS.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaPOMDP/AEMS.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaPOMDP/AEMS.jl?branch=master)

Implements anytime error minimization search (AEMS) solver for POMDPs.

# Installation


# Quick Use

```julia
solver = AEMSSolver()
```

# Solver Options

* `n_iterations` 
* `max_time`
* `updater`
* `lower_bound`
* `upper_bound`
* `root_manager`

# Visualization

```julia
# suppose you have a planner and have called action (to generate a tree)

using D3Trees

tree = D3Tree(planner)      # creates a visualization tree
inchrome(tree)              # opens chrome tab to show visualization tree
```
