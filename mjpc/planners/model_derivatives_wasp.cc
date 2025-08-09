//
// Created by dylan on 8/8/25.
//

#include "model_derivatives_wasp.h"


#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
namespace mjpc {
    void ModelDerivativesWASP::Allocate(int dim_state_derivative, int dim_action, int dim_sensor,
                      int T) {
        ModelDerivatives::Allocate(dim_state_derivative, dim_action, dim_sensor, T);
        needs_allocate_cache = true;
    }
    void ModelDerivativesWASP::Reset(int dim_state_derivative, int dim_action, int dim_sensor, int T) {
        ModelDerivatives::Reset(dim_state_derivative, dim_action, dim_sensor, T);
        needs_reset_cache = true;
    }
    void ModelDerivativesWASP::Compute(const mjModel *m, const std::vector<UniqueMjData> &data, const double *x, const double *u, const double *h, int dim_state, int dim_state_derivative, int dim_action, int dim_sensor, int T, double tol, int mode, ThreadPool &pool, int skip) {
        // allocate or reset wasp caches
        // note here in "na" a stands for "activation" (not "action"), a part of the state vector.
        // nu == dim_action
        int nv = m->nv, na = m->na, nu = m->nu;
        if (needs_allocate_cache) {
            // DyDx
            DyDq.Allocate(nv,dim_state,T);
            DyDv.Allocate(nv,dim_state,T);
            DyDa.Allocate(na,dim_state,T);
            DyDu.Allocate(nu,dim_state,T);
            // DsDx
            DsDq.Allocate(nv,dim_sensor,T);
            DsDv.Allocate(nv,dim_sensor,T);
            DsDa.Allocate(na,dim_sensor,T);
            DsDu.Allocate(nu,dim_sensor,T);
            needs_allocate_cache = false;
        }
        else if (needs_reset_cache) {
            // DyDx
            DyDq.Reset(nv,dim_state,T);
            DyDv.Reset(nv,dim_state,T);
            DyDa.Reset(na,dim_state,T);
            DyDu.Reset(nu,dim_state,T);
            // DsDx
            DsDq.Reset(nv,dim_sensor,T);
            DsDv.Reset(nv,dim_sensor,T);
            DsDa.Reset(na,dim_sensor,T);
            DsDu.Reset(nu,dim_sensor,T);
            needs_reset_cache = false;
        }
        ModelDerivatives::Compute(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor, T, tol, mode, pool, skip);
    }
    void ModelDerivativesWASP::OneStepDerivatives(const mjModel *m,
        const std::vector<UniqueMjData> &data,
        const double *x, const double *u, const double *h,
        int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
        int t, int T,
        double tol,
        int mode,
        ThreadPool &pool) {
        pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &DyDq = DyDq, &DyDv = DyDv, &DyDa= DyDa, &DyDu=DyDu,
               &DsDq = DsDq, &DsDv = DsDv, &DsDa= DsDa, &DsDu=DsDu,
               &x, &u, &h,
               dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
               mode, t, T]()
       {
          mjData* d = data[ThreadPool::WorkerId()].get();
          // set state
          SetState(m, d, x + t * dim_state);
          d->time = h[t];
          mjWASPCache DyDqLocal = DyDq.RawData(t), DyDvLocal = DyDv.RawData(t), DyDaLocal = DyDa.RawData(t), DyDuLocal= DyDu.RawData(t);
            mjWASPCache DsDqLocal = DsDq.RawData(t), DsDvLocal = DsDv.RawData(t), DsDaLocal = DsDa.RawData(t), DsDuLocal= DsDu.RawData(t);
          // set action
          mju_copy(d->ctrl, u + t * dim_action, dim_action);

          // Jacobians
          if (t == T - 1) {
            // Jacobians
            mjd_transitionWASP(m, d, tol, mode,
                              nullptr,
                              nullptr,
                             DataAt(C, t * (dim_sensor * dim_state_derivative)),
                             nullptr,
                             nullptr, nullptr, nullptr, nullptr,
                             &DsDqLocal, &DsDvLocal, &DsDaLocal,nullptr);
          } else {
            // derivatives
            mjd_transitionWASP(
                m, d, tol, mode,
                DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
                DataAt(B, t * (dim_state_derivative * dim_action)),
                DataAt(C, t * (dim_sensor * dim_state_derivative)),
                DataAt(D, t * (dim_sensor * dim_action)),
                &DyDqLocal, &DyDvLocal, &DyDaLocal, &DyDuLocal,
                &DsDqLocal, &DsDvLocal, &DsDaLocal, &DsDuLocal
                );
          }
});
    }





}