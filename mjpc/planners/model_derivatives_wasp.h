//
// Created by dylan on 8/8/25.
//

#ifndef MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_
#define MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_

#include "model_derivatives.h"
#include <mujoco/mujoco.h>
//#include <boost/circular_buffer.hpp>

#include <cstdlib>
#include <vector>

#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
namespace mjpc {
    class GradientPlanner;
    class iLQGPlanner;
    class ModelDerivativesWASP: public ModelDerivatives{
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

        // update cache n steps forward
        void RolloutCache(const mjModel* m,int n);
    private:
        void ParaDerivEval(const mjModel *m,
            const std::vector<UniqueMjData> &data,
            const double *x, const double *u, const double *h,
            int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
           int T,
            double tol,
            int mode,
            ThreadPool &pool) override;
        void ParaTDerivEval(const mjModel *m,
            const std::vector<UniqueMjData> &data,
            const double *x, const double *u, const double *h,
            int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
           int T,
            double tol,
            int mode,
            ThreadPool &pool);
        void ParaAllDerivEval(const mjModel *m,
            const std::vector<UniqueMjData> &data,
            const double *x, const double *u, const double *h,
            int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
         int T,
            double tol,
            int mode,
            ThreadPool &pool);
        void AllocateWASPData(const mjModel *m, int T);
        void HeuristicUpdate(double cost);
        void ResetWASPData(const mjModel *m, int T);
        mjWASPBasis *qv_basis=nullptr, *u_basis=nullptr, *a_basis=nullptr;
           std::vector<mjWASPCache*> DyDq, DyDv, DyDa;
     std::vector<mjWASPCache*> DyDu;
     std::vector<mjWASPCache*> DsDq, DsDv, DsDa;
        std::vector<mjWASPCache*> DsDu;
        bool needs_allocate_cache=true;
        bool needs_reset_cache=false;
        bool use_wasp_identity_basis=false;
        bool all_in_parallel=false;
        bool cache_rollout=true;
        // tuned interactively in planners' GUI
        friend class GradientPlanner;
        friend class iLQGPlanner;
        double x_eps=1e-4;
        double u_eps=1e-4;
        double max_x_eps = 1.0;
        double max_u_eps = 1.0;
        double T_eps=0.5;
        int q_max_wasp_iters=1;
        int v_max_wasp_iters=1;
        int a_max_wasp_iters=1;
        int u_max_wasp_iters=1;
        std::atomic<int> num_dynamics_called;
        bool heuristic_mode = true;

        std::vector<double> AT, BT, CT, DT;
    };
}



#endif // MJPC_PLANNERS_MODEL_DERIVATIVES_WASP_H_