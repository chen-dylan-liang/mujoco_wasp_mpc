//
// Created by dylan on 8/8/25.
//

#ifndef MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_
#define MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_

#include "model_derivatives.h"

namespace mjpc {

    struct mjpcWASPCache {
        std::vector<double> Delta_X;
        std::vector<double> C1;
        std::vector<double> C2;
        std::vector<double> F_hat;
        std::vector<double> fi;
        std::vector<int> i;
        void Allocate(int dim_inputs, int dim_outputs, int T) {
            Delta_X.resize(T*dim_inputs*dim_inputs);
            C1.resize((T*dim_inputs)*dim_outputs*dim_inputs);
            C2.resize((T*dim_inputs)*dim_inputs);
            F_hat.resize(T*dim_outputs*dim_inputs);
            fi.resize(T*dim_outputs);
            i.resize(T);
        }
        void Reset(int dim_inputs, int dim_outputs, int T) {
            std::fill(Delta_X.begin(), Delta_X.begin() + T*dim_inputs*dim_inputs, 0);
            std::fill(C1.begin(), C1.begin() + (T*dim_inputs)*dim_outputs*dim_inputs, 0);
            std::fill(C2.begin(), C2.begin() + (T*dim_inputs)*dim_inputs, 0);
            std::fill(F_hat.begin(), F_hat.begin() + T*dim_outputs*dim_inputs, 0);
            std::fill(fi.begin(), fi.begin() + T*dim_outputs, 0);
            std::fill(i.begin(), i.begin() + T, 0);
        }
    };

    class ModelDerivativesWASP: public ModelDerivatives {
    public:
        // constructor
        ModelDerivativesWASP() = default;

        // destructor
        ~ModelDerivativesWASP() = default;

        // compute derivatives at all time steps
        void Compute(const mjModel* m, const std::vector<UniqueMjData>& data,
               const double* x, const double* u, const double* h, int dim_state,
               int dim_state_derivative, int dim_action, int dim_sensor, int T,
               double tol, int mode, ThreadPool& pool, int skip = 0) override;
    private:
        mjpcWASPCache DyDq, DyDv, DyDa;
        mjpcWASPCache DyDu;
        mjpcWASPCache DsDq, DsDv, DsDa;
        mjpcWASPCache DsDu;
        bool needs_allocate_cache=true;
        bool needs_reset_cache=false;
    };
}



#endif // MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_