//
// Created by dylan on 8/8/25.
//

#include "model_derivatives_wasp.h"

namespace mjpc {
    void ModelDerivativesWASP::Compute(const mjModel *m, const std::vector<UniqueMjData> &data, const double *x, const double *u, const double *h, int dim_state, int dim_state_derivative, int dim_action, int dim_sensor, int T, double tol, int mode, ThreadPool &pool, int skip) {
        if (needs_allocate_cache) {

        }
        else if (needs_reset_cache) {

        }
    }


}