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
        DyDq.set_capacity(T);
        DyDq.set_capacity(T);
        DyDv.set_capacity(T);
        DyDa.set_capacity(T);
        DyDu.set_capacity(T);
        DsDq.set_capacity(T);
        DsDv.set_capacity(T);
        DsDa.set_capacity(T);
        DsDu.set_capacity(T);
        needs_allocate_cache = true;
    }
    void ModelDerivativesWASP::Reset(int dim_state_derivative, int dim_action, int dim_sensor, int T) {
        ModelDerivatives::Reset(dim_state_derivative, dim_action, dim_sensor, T);
        needs_reset_cache = true;
    }

    void ModelDerivativesWASP::RolloutCache(int n, int dim_v, int dim_a, int dim_u, int dim_y, int dim_s) {
        if (!needs_allocate_cache&&!needs_reset_cache) {
            // zero out the caches expired
            for (int i=0; i<n; i++) {
                mj_zeroWASPCache(DyDq[i], dim_v, dim_y);
                mj_zeroWASPCache(DyDv[i], dim_v, dim_y);
                mj_zeroWASPCache(DyDa[i], dim_a, dim_y);
                mj_zeroWASPCache(DyDu[i], dim_u, dim_y);
                mj_zeroWASPCache(DsDq[i], dim_v, dim_s);
                mj_zeroWASPCache(DsDv[i], dim_v, dim_s);
                mj_zeroWASPCache(DsDa[i], dim_a, dim_s);
                mj_zeroWASPCache(DsDu[i], dim_u, dim_s);
            }
            // go forward n steps
            for (int i=0; i<n; i++) {
                DyDq.push_back(DyDq.front());
                DyDv.push_back(DyDv.front());
                DyDa.push_back(DyDa.front());
                DyDu.push_back(DyDu.front());
                DsDq.push_back(DsDq.front());
                DsDv.push_back(DsDv.front());
                DsDa.push_back(DsDa.front());
                DsDu.push_back(DsDu.front());
            }
        }
    }

    void ModelDerivativesWASP::Compute(const mjModel *m, const std::vector<UniqueMjData> &data, const double *x, const double *u, const double *h, int dim_state, int dim_state_derivative, int dim_action, int dim_sensor, int T, double tol, int mode, ThreadPool &pool, int skip) {
        // allocate or reset wasp caches
        // note here in "na" a stands for "activation" (not "action"), a part of the state vector.
        // nu == dim_action
        int nv = m->nv, na = m->na, nu = m->nu;
        if (needs_allocate_cache) {
            for (int t=0; t<T; ++t) {
                DyDq.push_back(mj_newWASPCache(nv,dim_state_derivative,1,use_wasp_identity_basis));
                DyDv.push_back(mj_newWASPCache(nv,dim_state_derivative,1,use_wasp_identity_basis));
                DyDa.push_back(mj_newWASPCache(na,dim_state_derivative,1,use_wasp_identity_basis));
                DyDu.push_back(mj_newWASPCache(nu,dim_state_derivative,1,use_wasp_identity_basis));
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
                mj_zeroWASPCache(DyDq[t], nv,dim_state_derivative);
                mj_zeroWASPCache(DyDv[t], nv,dim_state_derivative);
                mj_zeroWASPCache(DyDa[t], na,dim_state_derivative);
                mj_zeroWASPCache(DyDu[t], nu,dim_state_derivative);
                mj_zeroWASPCache(DsDq[t], nv,dim_sensor);
                mj_zeroWASPCache(DsDv[t], nv,dim_sensor);
                mj_zeroWASPCache(DsDa[t], na,dim_sensor);
                mj_zeroWASPCache(DsDu[t], nu,dim_sensor);
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
          /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], this->DsDa[t], nullptr);
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
          /*DyDq..DyDu*/ this->DyDq[t],this->DyDv[t], this->DyDa[t], this->DyDu[t],
          /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], this->DsDa[t], this->DsDu[t]);
    }
  });

}






}