
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Copyright (c) 2025 Christian Fuellner <christian.fuellner@kit.edu>
################################################################################

function set_solver_for_model(
    model::SDDP.PolicyGraph,
    algo_params::LogLinearSDDP.AlgoParams,
)

    for (_, node) in model.nodes
        JuMP.set_attribute(node.subproblem, MOI.Silent(), algo_params.silent)
    end
end

# JuMP.set_optimizer(subproblem, JuMP.optimizer_with_attributes(() -> GAMS.Optimizer()))