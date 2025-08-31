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
        if (all_in_parallel) {
            AT.resize(dim_state_derivative * dim_state_derivative * T);
            BT.resize(dim_state_derivative * dim_action * T);
            CT.resize(dim_sensor * dim_state_derivative * T);
            DT.resize(dim_sensor * dim_action * T);
        }
        needs_allocate_cache = true;
    }

    void ModelDerivativesWASP::Reset(int dim_state_derivative, int dim_action, int dim_sensor, int T) {
        ModelDerivatives::Reset(dim_state_derivative, dim_action, dim_sensor, T);
        if (all_in_parallel) {
            std::fill(AT.begin(),
AT.begin() + T * dim_state_derivative * dim_state_derivative, 0.0);
            std::fill(BT.begin(), BT.begin() + T * dim_state_derivative * dim_action, 0.0);
            std::fill(CT.begin(), CT.begin() + T * dim_sensor * dim_state_derivative, 0.0);
            std::fill(DT.begin(), DT.begin() + T * dim_sensor * dim_action, 0.0);
        }
        needs_reset_cache = true;
    }

    void ModelDerivativesWASP::RolloutCache(int n, int dim_v, int dim_a, int dim_u, int dim_y, int dim_s) {
        if (!needs_allocate_cache && !needs_reset_cache) {
            // zero out the caches expired
            for (int i = 0; i < n; i++) {
                mj_zeroWASPCache(DyDq[i], dim_v, dim_y);
                mj_zeroWASPCache(DyDv[i], dim_v, dim_y);
                if (!DyDa.empty()) mj_zeroWASPCache(DyDa[i], dim_a, dim_y);
                mj_zeroWASPCache(DyDu[i], dim_u, dim_y);
                mj_zeroWASPCache(DsDq[i], dim_v, dim_s);
                mj_zeroWASPCache(DsDv[i], dim_v, dim_s);
                if (!DsDa.empty()) mj_zeroWASPCache(DsDa[i], dim_a, dim_s);
                mj_zeroWASPCache(DsDu[i], dim_u, dim_s);
            }
            // go forward n steps
            for (int i = 0; i < n; i++) {
                DyDq.push_back(DyDq.front());
                DyDv.push_back(DyDv.front());
                if (!DyDa.empty())DyDa.push_back(DyDa.front());
                DyDu.push_back(DyDu.front());
                DsDq.push_back(DsDq.front());
                DsDv.push_back(DsDv.front());
                if (!DsDa.empty()) DsDa.push_back(DsDa.front());
                DsDu.push_back(DsDu.front());
            }
        }
    }

    void ModelDerivativesWASP::Compute(const mjModel *m, const std::vector<UniqueMjData> &data, const double *x,
                                       const double *u, const double *h, int dim_state, int dim_state_derivative,
                                       int dim_action, int dim_sensor, int T, double tol, int mode, ThreadPool &pool,
                                       int skip) {
        // allocate or reset wasp caches
        // note here in "na" a stands for "activation" (not "action"), a part of the state vector.
        // nu == dim_action
        int nv = m->nv, na = m->na, nu = m->nu;
        if (needs_allocate_cache) {
            // clear
            DyDq.clear();
            DyDv.clear();
            DyDa.clear();
            DyDu.clear();
            DsDq.clear();
            DsDv.clear();
            DsDa.clear();
            DsDu.clear();
            // reset capacity and re-push
            DyDq.set_capacity(T);
            DyDv.set_capacity(T);
            if (m->na > 0) DyDa.set_capacity(T);
            else DyDa.set_capacity(0);
            DyDu.set_capacity(T);
            DsDq.set_capacity(T);
            DsDv.set_capacity(T);
            if (m->na > 0) DsDa.set_capacity(T);
            else DsDa.set_capacity(0);
            DsDu.set_capacity(T);
            for (int t = 0; t < T; ++t) {
                DyDq.push_back(mj_newWASPCache(nv, dim_state_derivative, 1, use_wasp_identity_basis));
                DyDv.push_back(mj_newWASPCache(nv, dim_state_derivative, 1, use_wasp_identity_basis));
                if (m->na > 0) DyDa.push_back(mj_newWASPCache(na, dim_state_derivative, 1, use_wasp_identity_basis));
                DyDu.push_back(mj_newWASPCache(nu, dim_state_derivative, 1, use_wasp_identity_basis));
                // Ds caches use the same bases as these of Dy caches
                DsDq.push_back(mj_newWASPCache(nv, dim_sensor, 0, use_wasp_identity_basis));
                DsDv.push_back(mj_newWASPCache(nv, dim_sensor, 0, use_wasp_identity_basis));
                if (m->na > 0) DsDa.push_back(mj_newWASPCache(na, dim_sensor, 0, use_wasp_identity_basis));
                DsDu.push_back(mj_newWASPCache(nu, dim_sensor, 0, use_wasp_identity_basis));
                mj_copyWASPCacheBasis(DsDq[t], DyDq[t], nv);
                mj_copyWASPCacheBasis(DsDv[t], DyDv[t], nv);
                if (m->na > 0) mj_copyWASPCacheBasis(DsDa[t], DyDa[t], na);
                mj_copyWASPCacheBasis(DsDu[t], DyDu[t], nu);
            }
            needs_allocate_cache = false;
        } else if (needs_reset_cache) {
            for (int t = 0; t < T; ++t) {
                mj_zeroWASPCache(DyDq[t], nv, dim_state_derivative);
                mj_zeroWASPCache(DyDv[t], nv, dim_state_derivative);
                if (m->na > 0) mj_zeroWASPCache(DyDa[t], na, dim_state_derivative);
                mj_zeroWASPCache(DyDu[t], nu, dim_state_derivative);
                mj_zeroWASPCache(DsDq[t], nv, dim_sensor);
                mj_zeroWASPCache(DsDv[t], nv, dim_sensor);
                if (m->na > 0) mj_zeroWASPCache(DsDa[t], na, dim_sensor);
                mj_zeroWASPCache(DsDu[t], nu, dim_sensor);
            }
            needs_reset_cache = false;
        }
        ModelDerivatives::Compute(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor, T, tol,
                                  mode, pool, skip);
    }

    void ModelDerivativesWASP::ParaDerivEval(
        const mjModel *m,
        const std::vector<UniqueMjData> &data,
        const double *x, const double *u, const double *h,
        int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
        int T, double tol, int mode, ThreadPool &pool) {
        if (!all_in_parallel)
            ParaTDerivEval(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor,
                           T, tol, mode, pool);
        else
            ParaAllDerivEval(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor,
                             T, tol, mode, pool);
    }

    void ModelDerivativesWASP::ParaTDerivEval(
        const mjModel *m,
        const std::vector<UniqueMjData> &data,
        const double *x, const double *u, const double *h,
        int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
        int T, double tol, int mode, ThreadPool &pool) {
        int count_before = pool.GetCount();
        for (int t: evaluate_) {
            // Stable pointer to the vector; avoids copying and avoids dangling ref-to-param.
            auto data_ptr = &data;

            pool.Schedule([this, // access A,B,C,D, Dy*, Ds*, q_*, v_*, a_*, u_*...
                    m, // copy pointer
                    data_ptr, // copy pointer to vector
                    x, u, h, // copy pointers
                    dim_state, dim_state_derivative, dim_action, dim_sensor,
                    tol, mode, t, T]() // copy small scalars
                {
                    mjData *d = (*data_ptr)[ThreadPool::WorkerId()].get();
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
                            /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], m->na > 0 ? this->DsDa[t] : nullptr, nullptr);
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
                            /*DyDq..DyDu*/ this->DyDq[t], this->DyDv[t], m->na > 0 ? this->DyDa[t] : nullptr,
                            this->DyDu[t],
                            /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], m->na > 0 ? this->DsDa[t] : nullptr,
                            this->DsDu[t]);
                    }
                });
        }
        pool.WaitCount(count_before + evaluate_.size());
        pool.ResetCount();
    }

    void ModelDerivativesWASP::ParaAllDerivEval(
        const mjModel *m,
        const std::vector<UniqueMjData> &data,
        const double *x, const double *u, const double *h,
        int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
        int T, double tol, int mode, ThreadPool &pool) {
        if (m->na > 0) {
            int count_before = pool.GetCount();
            for (int t: evaluate_)
                for (int type = 0; type < 8; ++type) {
                    // Stable pointer to the vector; avoids copying and avoids dangling ref-to-param.
                    auto data_ptr = &data;

                    pool.Schedule([this, // access A,B,C,D, Dy*, Ds*, q_*, v_*, a_*, u_*...
                            m, // copy pointer
                            data_ptr, // copy pointer to vector
                            x, u, h, // copy pointers
                            dim_state, dim_state_derivative, dim_action, dim_sensor,
                            tol, mode, t, T, type]() // copy small scalars
                        {
                            mjData *d = (*data_ptr)[ThreadPool::WorkerId()].get();
                            // set state/time
                            SetState(m, d, x + t * dim_state);
                            d->time = h[t];

                            // set action
                            mju_copy(d->ctrl, u + t * dim_action, dim_action);

                            if (t == T - 1) {
                                switch (type) {
                                    // DsDq
                                    case 4:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 5:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDa
                                    case 6:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->a_dtheta, this->a_dell,
                                                                    this->a_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + 2 *
                                                                           m->nv * dim_sensor), this->DsDa[t], mjDsDa);
                                        break;
                                    // DsDu
                                    case 7:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                    // DyDx
                                    default:
                                        break;
                                }
                            } else {
                                switch (type) {
                                    // DyDq
                                    case 0:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative)), this->DyDq[t],
                                                                    mjDyDq);
                                        break;
                                    // DyDv
                                    case 1:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + m->nv *
                                                                           dim_state_derivative), this->DyDv[t],
                                                                    mjDyDv);
                                        break;
                                    // DyDa
                                    case 2:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->a_dtheta, this->a_dell,
                                                                    this->a_max_wasp_iters,
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + 2 * m->nv *
                                                                           dim_state_derivative), this->DyDa[t],
                                                                    mjDyDa);
                                        break;
                                    // DyDu
                                    case 3:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->BT,
                                                                           t * (dim_state_derivative * dim_action)),
                                                                    this->DyDu[t], mjDyDu);
                                        break;
                                    // DsDq
                                    case 4:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 5:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDa
                                    case 6:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->a_dtheta, this->a_dell,
                                                                    this->a_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + 2 *
                                                                           m->nv * dim_sensor), this->DsDa[t], mjDsDa);
                                        break;
                                    // DsDu
                                    default:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                }
                            }
                        });
                }
            pool.WaitCount(count_before + evaluate_.size() * 8);
            pool.ResetCount();
        } else {
            int count_before = pool.GetCount();
            for (int t: evaluate_)
                for (int type = 0; type < 6; ++type) {
                    // Stable pointer to the vector; avoids copying and avoids dangling ref-to-param.
                    auto data_ptr = &data;

                    pool.Schedule([this, // access A,B,C,D, Dy*, Ds*, q_*, v_*, a_*, u_*...
                            m, // copy pointer
                            data_ptr, // copy pointer to vector
                            x, u, h, // copy pointers
                            dim_state, dim_state_derivative, dim_action, dim_sensor,
                            tol, mode, t, T, type]() // copy small scalars
                        {
                            mjData *d = (*data_ptr)[ThreadPool::WorkerId()].get();
                            // set state/time
                            SetState(m, d, x + t * dim_state);
                            d->time = h[t];

                            // set action
                            mju_copy(d->ctrl, u + t * dim_action, dim_action);

                            if (t == T - 1) {
                                switch (type) {
                                    // DsDq
                                    case 3:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 4:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDu
                                    case 5:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                    // DyDx
                                    default:
                                        break;
                                }
                            } else {
                                switch (type) {
                                    // DyDq
                                    case 0:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative)), this->DyDq[t],
                                                                    mjDyDq);
                                        break;
                                    // DyDv
                                    case 1:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + m->nv *
                                                                           dim_state_derivative), this->DyDv[t],
                                                                    mjDyDv);
                                        break;
                                    // DyDu
                                    case 2:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->BT,
                                                                           t * (dim_state_derivative * dim_action)),
                                                                    this->DyDu[t], mjDyDu);
                                        break;
                                    // DsDq
                                    case 3:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->q_dtheta, this->q_dell,
                                                                    this->q_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 4:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->v_dtheta, this->v_dell,
                                                                    this->v_max_wasp_iters,
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDu
                                    default:
                                        mjd_transitionWASPOneThread(m, d, tol, mode,
                                                                    this->u_dtheta, this->u_dell,
                                                                    this->u_max_wasp_iters,
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                }
                            }
                        });
                }
            pool.WaitCount(count_before + evaluate_.size() * 6);
            pool.ResetCount();
        }
        int count_before = pool.GetCount();
        for (int t: evaluate_) {
            pool.Schedule([this, // access A,B,C,D, Dy*, Ds*, q_*, v_*, a_*, u_*...
                         dim_state_derivative, dim_action, dim_sensor,
                      t]() // copy small scalars
            {
                mju_transpose(DataAt(this->A, t*dim_state_derivative*dim_state_derivative), DataAt(this->AT, t*dim_state_derivative*dim_state_derivative),
                             dim_state_derivative, dim_state_derivative);
                mju_transpose(DataAt(this->B, t*dim_action*dim_state_derivative), DataAt(this->BT, t*dim_action*dim_state_derivative),
                       dim_action, dim_state_derivative);
                mju_transpose(DataAt(this->C, t*dim_state_derivative*dim_sensor), DataAt(this->CT, t*dim_state_derivative*dim_sensor),
                       dim_state_derivative, dim_sensor);
                mju_transpose(DataAt(this->D, t*dim_action*dim_sensor), DataAt(this->DT, t*dim_action*dim_sensor),
                       dim_action, dim_sensor);
            });
        }
        pool.WaitCount(count_before + evaluate_.size());
        pool.ResetCount();
    }
}
