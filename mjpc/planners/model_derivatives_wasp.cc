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
                DyDq.push_back(mj_newWASPCache(nv,dim_state,1,use_wasp_identity_basis));
                DyDv.push_back(mj_newWASPCache(nv,dim_state,1,use_wasp_identity_basis));
                DyDa.push_back(mj_newWASPCache(na,dim_state,1,use_wasp_identity_basis));
                DyDu.push_back(mj_newWASPCache(nu,dim_state,1,use_wasp_identity_basis));
                // Ds caches use the same bases as these of Dy caches
                DsDq.push_back(mj_newWASPCache(nv,dim_sensor,0,use_wasp_identity_basis));
                DsDv.push_back(mj_newWASPCache(nv,dim_sensor,0,use_wasp_identity_basis));
                DsDa.push_back(mj_newWASPCache(na,dim_sensor,0,use_wasp_identity_basis));
                DsDu.push_back(mj_newWASPCache(nu,dim_sensor,0,use_wasp_identity_basis));
                mj_copyWASPCacheBasis(DsDq[t], DyDq[t], nv);
                mj_copyWASPCacheBasis(DsDv[t], DyDv[t], nv);
                mj_copyWASPCacheBasis(DsDa[t], DyDa[t], na);
                mj_copyWASPCacheBasis(DsDu[t], DyDu[t], nu);
            }
            needs_allocate_cache = false;
        }
        else if (needs_reset_cache) {
            for (int t=0; t<T; ++t) {
                mj_zeroWASPCache(DyDq[t], 0,dim_state);
                mj_zeroWASPCache(DyDv[t], 0,dim_state);
                mj_zeroWASPCache(DyDa[t], 0,dim_state);
                mj_zeroWASPCache(DyDu[t], 0,dim_state);
                mj_zeroWASPCache(DsDq[t], 0,dim_sensor);
                mj_zeroWASPCache(DsDv[t], 0,dim_sensor);
                mj_zeroWASPCache(DsDa[t], 0,dim_sensor);
                mj_zeroWASPCache(DsDu[t], 0,dim_sensor);
            }
            needs_reset_cache = false;
        }
        ModelDerivatives::Compute(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor, T, tol, mode, pool, skip);
    }

    void ModelDerivativesWASP::OneStepDerivatives(
    const mjModel* m,
    const std::vector<UniqueMjData>& data,
    const double* x, const double* u, const double* h,
    int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
    int t, int T, double tol, int mode, ThreadPool& pool) {

  // Stable pointer to the vector; avoids copying and avoids dangling ref-to-param.
  auto data_ptr = &data;

  pool.Schedule([this,               // access A,B,C,D, Dy*, Ds*, q_*, v_*, a_*, u_*...
                 m,                  // copy pointer
                 data_ptr,           // copy pointer to vector
                 x, u, h,            // copy pointers
                 dim_state, dim_state_derivative, dim_action, dim_sensor,
                 tol, mode, t, T]()  // copy small scalars
  {
    mjData* d = (*data_ptr)[ThreadPool::WorkerId()].get();

    // set state/time
    SetState(m, d, x + t * dim_state);
    d->time = h[t];

    // set action
    mju_copy(d->ctrl, u + t * dim_action, dim_action);

    // local aliases to per-t caches (members)
    mjWASPCache* DyDqLocal = this->DyDq[t];
    mjWASPCache* DyDvLocal = this->DyDv[t];
    mjWASPCache* DyDaLocal = this->DyDa[t];
    mjWASPCache* DyDuLocal = this->DyDu[t];
    mjWASPCache* DsDqLocal = this->DsDq[t];
    mjWASPCache* DsDvLocal = this->DsDv[t];
    mjWASPCache* DsDaLocal = this->DsDa[t];
    mjWASPCache* DsDuLocal = this->DsDu[t];

    if (t == T - 1) {
      mjd_transitionWASP(
          m, d, tol, mode,
          this->q_dtheta, this->q_dell, this->q_max_wasp_iters,
          this->v_dtheta, this->v_dell, this->v_max_wasp_iters,
          this->a_dtheta, this->a_dell, this->a_max_wasp_iters,
          this->u_dtheta, this->u_dell, this->u_max_wasp_iters,
          /*A*/ nullptr,
          /*B*/ nullptr,
          /*C*/ DataAt(this->C, t * (dim_sensor * dim_state_derivative)),
          /*D*/ nullptr,
          /*DyDq..DyDu*/ nullptr, nullptr, nullptr, nullptr,
          /*DsDq..DsDu*/ DsDqLocal, DsDvLocal, DsDaLocal, nullptr);
    } else {
      mjd_transitionWASP(
          m, d, tol, mode,
          this->q_dtheta, this->q_dell, this->q_max_wasp_iters,
          this->v_dtheta, this->v_dell, this->v_max_wasp_iters,
          this->a_dtheta, this->a_dell, this->a_max_wasp_iters,
          this->u_dtheta, this->u_dell, this->u_max_wasp_iters,
          /*A*/ DataAt(this->A, t * (dim_state_derivative * dim_state_derivative)),
          /*B*/ DataAt(this->B, t * (dim_state_derivative * dim_action)),
          /*C*/ DataAt(this->C, t * (dim_sensor * dim_state_derivative)),
          /*D*/ DataAt(this->D, t * (dim_sensor * dim_action)),
          /*DyDq..DyDu*/ DyDqLocal, DyDvLocal, DyDaLocal, DyDuLocal,
          /*DsDq..DsDu*/ DsDqLocal, DsDvLocal, DsDaLocal, DsDuLocal);
    }
  });
}






}