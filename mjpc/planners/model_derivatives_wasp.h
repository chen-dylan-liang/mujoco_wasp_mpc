//
// Created by dylan on 8/8/25.
//

#ifndef MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_
#define MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_

#include "model_derivatives.h"
#include <mujoco/mujoco.h>

#include <cstdlib>
#include <vector>

#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
namespace mjpc {

    class ModelDerivativesWASP: public ModelDerivatives {
    public:
        // constructor
        ModelDerivativesWASP() = default;

        // destructor
        ~ModelDerivativesWASP() {
            for (int i=0; i<DyDq.size();i++) {
                mj_deleteWASPCache(DyDq[i]);
                DyDq[i]=nullptr;
            }
            for (int i=0; i<DyDv.size();i++) {
                mj_deleteWASPCache(DyDv[i]);
                DyDv[i]=nullptr;
            }
            for (int i=0; i<DyDa.size();i++) {
                mj_deleteWASPCache(DyDa[i]);
                DyDa[i]=nullptr;
            }
            for (int i=0; i<DyDu.size();i++) {
                mj_deleteWASPCache(DyDu[i]);
                DyDu[i]=nullptr;
            }
            for (int i=0; i<DsDq.size();i++) {
                mj_deleteWASPCache(DsDq[i]);
                DsDq[i]=nullptr;
            }
            for (int i=0; i<DsDv.size();i++) {
                mj_deleteWASPCache(DsDv[i]);
                DsDv[i]=nullptr;
            }
            for (int i=0; i<DsDa.size();i++) {
                mj_deleteWASPCache(DsDa[i]);
                DsDa[i]=nullptr;
            }
            for (int i=0; i<DsDu.size();i++) {
                mj_deleteWASPCache(DsDu[i]);
                DsDu[i]=nullptr;
            }
        }

        // allocate memory
        void Allocate(int dim_state_derivative, int dim_action, int dim_sensor,
                      int T) override;

        // reset memory to zeros
        void Reset(int dim_state_derivative, int dim_action, int dim_sensor, int T) override;

        // compute derivatives at all time steps
        void Compute(const mjModel* m, const std::vector<UniqueMjData>& data,
               const double* x, const double* u, const double* h, int dim_state,
               int dim_state_derivative, int dim_action, int dim_sensor, int T,
               double tol, int mode, ThreadPool& pool, int skip = 0) override;
    private:
        void OneStepDerivatives(const mjModel *m,
            const std::vector<UniqueMjData> &data,
            const double *x, const double *u, const double *h,
            int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
            int t, int T,
            double tol,
            int mode,
            ThreadPool &pool) override;
        std::vector<mjWASPCache*> DyDq, DyDv, DyDa;
        std::vector<mjWASPCache*> DyDu;
        std::vector<mjWASPCache*> DsDq, DsDv, DsDa;
        std::vector<mjWASPCache*> DsDu;
        bool needs_allocate_cache=true;
        bool needs_reset_cache=false;
        bool use_wasp_identity_basis=false;
        double dtheta=1e10, dell=1e-10;
        int max_wasp_iters=5;
    };
}



#endif // MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_