module AEMS

using POMDPs
using POMDPToolbox

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater

export
    AEMSSolver,
    AEMSPlanner,
    solve,
    action,
    value

include("vanilla.jl")
include("visualization.jl")
export visualize

end # module
