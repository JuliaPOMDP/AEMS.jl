module AEMS

using POMDPs
using POMDPToolbox

using FIB: FIBSolver

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater

export
    AEMSSolver,
    AEMSPlanner,
    solve,
    action,
    value,

    clear_graph,
    count_nodes,
    update_root

include("graph.jl")
include("bounds.jl")

include("vanilla.jl")

include("visualization.jl")

export visualize

end # module
