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
        DyDq.reserve(T);
        DyDv.reserve(T);
        DyDa.reserve(T);
        DyDu.reserve(T);
        DsDq.reserve(T);
        DsDv.reserve(T);
        DsDa.reserve(T);
        DsDu.reserve(T);
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
            for (int t=0; t<T; ++t) {
                DyDq.push_back(mj_newWASPCache(nv,dim_state,use_wasp_identity_basis));
                DyDv.push_back(mj_newWASPCache(nv,dim_state,use_wasp_identity_basis));
                DyDa.push_back(mj_newWASPCache(na,dim_state,use_wasp_identity_basis));
                DyDu.push_back(mj_newWASPCache(nu,dim_state,use_wasp_identity_basis));
                DsDq.push_back(mj_newWASPCache(nv,dim_sensor,use_wasp_identity_basis));
                DsDv.push_back(mj_newWASPCache(nv,dim_sensor,use_wasp_identity_basis));
                DsDa.push_back(mj_newWASPCache(na,dim_sensor,use_wasp_identity_basis));
                DsDu.push_back(mj_newWASPCache(nu,dim_sensor,use_wasp_identity_basis));
            }
            needs_allocate_cache = false;
        }
        else if (needs_reset_cache) {
            for (int t=0; t<T; ++t) {
                mj_resetWASPCache(DyDq[t], nv,dim_state,use_wasp_identity_basis);
                mj_resetWASPCache(DyDv[t], nv,dim_state,use_wasp_identity_basis);
                mj_resetWASPCache(DyDa[t], na,dim_state,use_wasp_identity_basis);
                mj_resetWASPCache(DyDu[t], nu,dim_state,use_wasp_identity_basis);
                mj_resetWASPCache(DsDq[t], nv,dim_sensor,use_wasp_identity_basis);
                mj_resetWASPCache(DsDv[t], nv,dim_sensor,use_wasp_identity_basis);
                mj_resetWASPCache(DsDa[t], na,dim_sensor,use_wasp_identity_basis);
                mj_resetWASPCache(DsDu[t], nu,dim_sensor,use_wasp_identity_basis);
            }
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
               &DsDq = DsDq, &DsDv = DsDv, &DsDa= DsDa, &DsDu=DsDu, dtheta=dtheta, dell=dell, max_wasp_iters=max_wasp_iters,
               &x, &u, &h,
               dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
               mode, t, T]()
       {
          mjData* d = data[ThreadPool::WorkerId()].get();
          // set state
          SetState(m, d, x + t * dim_state);
          d->time = h[t];
          mjWASPCache *DyDqLocal = DyDq[t], *DyDvLocal = DyDv[t], *DyDaLocal = DyDa[t], *DyDuLocal= DyDu[t];
            mjWASPCache *DsDqLocal = DsDq[t], *DsDvLocal = DsDv[t], *DsDaLocal = DsDa[t], *DsDuLocal= DsDu[t];
          // set action
          mju_copy(d->ctrl, u + t * dim_action, dim_action);

          // Jacobians
          if (t == T - 1) {
            // Jacobians
            mjd_transitionWASP(m, d, tol, mode,
                               dtheta, dell, max_wasp_iters,
                               dtheta, dell, max_wasp_iters,
                               dtheta, dell, max_wasp_iters,
                               dtheta, dell, max_wasp_iters,
                              nullptr,
                              nullptr,
                             DataAt(C, t * (dim_sensor * dim_state_derivative)),
                             nullptr,
                             nullptr, nullptr, nullptr, nullptr,
                             DsDqLocal, DsDvLocal, DsDaLocal,nullptr);
          } else {
            // derivatives
            mjd_transitionWASP(
                m, d, tol, mode,
                dtheta, dell, max_wasp_iters,
                dtheta, dell, max_wasp_iters,
                dtheta, dell, max_wasp_iters,
                dtheta, dell, max_wasp_iters,
                DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
                DataAt(B, t * (dim_state_derivative * dim_action)),
                DataAt(C, t * (dim_sensor * dim_state_derivative)),
                DataAt(D, t * (dim_sensor * dim_action)),
                DyDqLocal, DyDvLocal, DyDaLocal, DyDuLocal,
                DsDqLocal, DsDvLocal, DsDaLocal, DsDuLocal
                );
          }
});
    }





}