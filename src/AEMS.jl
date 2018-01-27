module AEMS

using POMDPs
using POMDPToolbox

using FIB: FIBSolver

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import POMDPs: simulate

export
    AEMSSolver,
    AEMSPlanner,
    solve,
    action,
    value,

    clear_graph!,
    count_nodes,
    update_root

include("graph.jl")
include("bounds.jl")

include("solver.jl")
include("action.jl")

include("simulate.jl")

include("visualization.jl")

export visualize

end # module
