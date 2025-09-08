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
        DyDq.resize(T);
        DyDv.resize(T);
        DyDa.resize(T);
        DyDa.resize(T);
        DyDu.resize(T);
        DsDq.resize(T);
        DsDv.resize(T);
        DsDa.resize(T);
        DsDa.resize(T);
        DsDu.resize(T);
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

    void ModelDerivativesWASP::RolloutCache(const mjModel* m, int n) {
        if (!needs_allocate_cache && !needs_reset_cache && cache_rollout) {
            // go forward n steps
            for (int i = 0; i < n; i++)
                if (i+n < DyDq.size()){
                mj_copyWASPCache(DyDq[i],DyDq[i+n], m->nv, 2*m->nv+m->na);
                     mj_copyWASPCache(DyDv[i],DyDv[i+n], m->nv, 2*m->nv+m->na);
                    if (!DyDa.empty()) mj_copyWASPCache(DyDa[i],DyDa[i+n], m->na, 2*m->nv+m->na);
                     mj_copyWASPCache(DyDu[i],DyDu[i+n],m->nu, 2*m->nv+m->na);
                     mj_copyWASPCache(DsDq[i],DsDq[i+n], m->nv, m->nsensordata);
                     mj_copyWASPCache(DsDv[i],DsDv[i+n], m->nv, m->nsensordata);
                    if (!DsDa.empty()) mj_copyWASPCache(DsDa[i],DsDa[i+n], m->na, m->nsensordata);
                     mj_copyWASPCache(DsDu[i],DsDu[i+n],m->nu, m->nsensordata);
            }
        }
    }
    void ModelDerivativesWASP::AllocateWASPData(const mjModel *m, int T) {
        if (needs_allocate_cache) {
            int nv = m->nv, na = m->na, nu = m->nu;
            int dim_state_derivative=2*nv+na, dim_sensor=m->nsensordata;
            // clear
            mj_deleteWASPBasis(qv_basis);
            qv_basis=nullptr;
            mj_deleteWASPBasis(a_basis);
            a_basis=nullptr;
            mj_deleteWASPBasis(u_basis);
            u_basis=nullptr;
            for (int t=0; t < DyDq.size(); ++t) {
                mj_deleteWASPCache(DyDq[t]);
                DyDq[t]=nullptr;
                mj_deleteWASPCache(DyDv[t]);
                DyDv[t]=nullptr;
                if (DyDa.size()) {
                    mj_deleteWASPCache(DyDa[t]);
                    DyDa[t]=nullptr;
                }
                   mj_deleteWASPCache(DyDu[t]);
                DyDu[t]=nullptr;
                mj_deleteWASPCache(DsDq[t]);
                DsDq[t]=nullptr;
                mj_deleteWASPCache(DsDv[t]);
                DsDv[t]=nullptr;
                if (DsDa.size()) {
                    mj_deleteWASPCache(DsDa[t]);
                    DsDa[t]=nullptr;
                }
                mj_deleteWASPCache(DsDu[t]);
                DsDu[t]=nullptr;
            }
            DyDq.clear();
            DyDv.clear();
            DyDa.clear();
            DyDu.clear();
            DsDq.clear();
            DsDv.clear();
            DsDa.clear();
            DsDu.clear();
            // reset
            qv_basis=mj_newWASPBasis(nv, use_wasp_identity_basis);
            u_basis=mj_newWASPBasis(nu, use_wasp_identity_basis);
            if (m->na>0) a_basis=mj_newWASPBasis(na, use_wasp_identity_basis);
            for (int t = 0; t < T; ++t) {
                DyDq.push_back(mj_newWASPCache(nv, dim_state_derivative));
                //DyDq[t]->i = t%nv;
                DyDv.push_back(mj_newWASPCache(nv, dim_state_derivative));
                //DyDv[t]->i = t%nv;
                if (m->na > 0) {
                    DyDa.push_back(mj_newWASPCache(na, dim_state_derivative));
                    //DyDa[t]->i = t%na;
                }
                DyDu.push_back(mj_newWASPCache(nu, dim_state_derivative));
                //DyDu[t]->i = t%nu;
                // Ds caches use the same bases as these of Dy caches
                DsDq.push_back(mj_newWASPCache(nv, dim_sensor));
                //DsDq[t]->i = t%nv;
                DsDv.push_back(mj_newWASPCache(nv, dim_sensor));
               // DsDv[t]->i = t%nv;
                if (m->na > 0) {
                    DsDa.push_back(mj_newWASPCache(na, dim_sensor));
                    //DsDa[t]->i = t%na;
                }
                DsDu.push_back(mj_newWASPCache(nu, dim_sensor));
               // DsDu[t]->i = t%nu;
            }
            needs_allocate_cache = false;
        }
    }

    void ModelDerivativesWASP::ResetWASPData(const mjModel *m, int T) {
        if (needs_reset_cache) {
            int nv = m->nv, na = m->na, nu = m->nu;
            int dim_state_derivative=2*nv+na, dim_sensor=m->nsensordata;
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
    }


    void ModelDerivativesWASP::Compute(const mjModel *m, const std::vector<UniqueMjData> &data, const double *x,
                                       const double *u, const double *h, int dim_state, int dim_state_derivative,
                                       int dim_action, int dim_sensor, int T, double tol, int mode, ThreadPool &pool,
                                       int skip) {
        AllocateWASPData(m,T);
        ResetWASPData(m,T);
        ModelDerivatives::Compute(m, data, x, u, h, dim_state, dim_state_derivative, dim_action, dim_sensor, T, tol,
                                  mode, pool, skip);
    }

    void ModelDerivativesWASP::ParaDerivEval(
        const mjModel *m,
        const std::vector<UniqueMjData> &data,
        const double *x, const double *u, const double *h,
        int dim_state, int dim_state_derivative, int dim_action, int dim_sensor,
        int T, double tol, int mode, ThreadPool &pool) {
        num_dynamics_called=0;
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
                    int cnt=0;
                    if (t == T - 1) {
                        cnt = mjd_transitionWASP(
                            m, this->qv_basis, this->qv_basis, this->a_basis,nullptr,
                            d, tol, mode,
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->nv),
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->nv),
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->na),
                            this->u_eps, this->u_eps, (int)(this->u_frac_wasp*m->nu),
                            /*A*/ nullptr,
                            /*B*/ nullptr,
                            /*C*/ DataAt(this->C, t * (dim_sensor * dim_state_derivative)),
                            /*D*/ nullptr,
                            /*DyDq..DyDu*/ nullptr, nullptr, nullptr, nullptr,
                            /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], m->na > 0 ? this->DsDa[t] : nullptr, nullptr);
                    } else {
                         cnt= mjd_transitionWASP(
                            m, this->qv_basis, this->qv_basis, this->a_basis,this->u_basis,
                            d, tol, mode,
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->nv),
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->nv),
                            this->x_eps, this->x_eps, (int)(this->x_frac_wasp*m->na),
                            this->u_eps, this->u_eps, (int)(this->u_frac_wasp*m->nu),
                            /*A*/ DataAt(this->A, t * (dim_state_derivative * dim_state_derivative)),
                            /*B*/ DataAt(this->B, t * (dim_state_derivative * dim_action)),
                            /*C*/ DataAt(this->C, t * (dim_sensor * dim_state_derivative)),
                            /*D*/ DataAt(this->D, t * (dim_sensor * dim_action)),
                            /*DyDq..DyDu*/ this->DyDq[t], this->DyDv[t], m->na > 0 ? this->DyDa[t] : nullptr,
                            this->DyDu[t],
                            /*DsDq..DsDu*/ this->DsDq[t], this->DsDv[t], m->na > 0 ? this->DsDa[t] : nullptr,
                            this->DsDu[t]);
                    }
                    num_dynamics_called+=cnt;
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
                            int cnt=0;
                            if (t == T - 1) {
                                switch (type) {
                                    // DsDq
                                    case 4:
                                         cnt=mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 5:
                                         cnt=mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDa
                                    case 6:
                                         cnt=mjd_transitionWASPOneThread(m, this->a_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->na),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + 2 *
                                                                           m->nv * dim_sensor), this->DsDa[t], mjDsDa);
                                        break;
                                    // DsDu
                                    case 7:
                                        cnt= mjd_transitionWASPOneThread(m, this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
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
                                        cnt= mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative)), this->DyDq[t],
                                                                    mjDyDq);
                                        break;
                                    // DyDv
                                    case 1:
                                        cnt= mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + m->nv *
                                                                           dim_state_derivative), this->DyDv[t],
                                                                    mjDyDv);
                                        break;
                                    // DyDa
                                    case 2:
                                        cnt= mjd_transitionWASPOneThread(m, this->a_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->na),
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + 2 * m->nv *
                                                                           dim_state_derivative), this->DyDa[t],
                                                                    mjDyDa);
                                        break;
                                    // DyDu
                                    case 3:
                                        cnt= mjd_transitionWASPOneThread(m,this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
                                                                    DataAt(this->BT,
                                                                           t * (dim_state_derivative * dim_action)),
                                                                    this->DyDu[t], mjDyDu);
                                        break;
                                    // DsDq
                                    case 4:
                                        cnt= mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 5:
                                        cnt= mjd_transitionWASPOneThread(m,this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDa
                                    case 6:
                                        cnt= mjd_transitionWASPOneThread(m,this->a_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->na),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + 2 *
                                                                           m->nv * dim_sensor), this->DsDa[t], mjDsDa);
                                        break;
                                    // DsDu
                                    default:
                                        cnt= mjd_transitionWASPOneThread(m, this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                }
                            }
                            num_dynamics_called+=cnt;
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
                            int cnt=0;
                            if (t == T - 1) {
                                switch (type) {
                                    // DsDq
                                    case 3:
                                        cnt= mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 4:
                                        cnt= mjd_transitionWASPOneThread(m,this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDu
                                    case 5:
                                        cnt= mjd_transitionWASPOneThread(m, this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
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
                                        cnt= mjd_transitionWASPOneThread(m,this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative)), this->DyDq[t],
                                                                    mjDyDq);
                                        break;
                                    // DyDv
                                    case 1:
                                        cnt= mjd_transitionWASPOneThread(m,this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->AT,
                                                                           t * (dim_state_derivative *
                                                                               dim_state_derivative) + m->nv *
                                                                           dim_state_derivative), this->DyDv[t],
                                                                    mjDyDv);
                                        break;
                                    // DyDu
                                    case 2:
                                        cnt= mjd_transitionWASPOneThread(m, this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
                                                                    DataAt(this->BT,
                                                                           t * (dim_state_derivative * dim_action)),
                                                                    this->DyDu[t], mjDyDu);
                                        break;
                                    // DsDq
                                    case 3:
                                        cnt= mjd_transitionWASPOneThread(m,this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative)),
                                                                    this->DsDq[t], mjDsDq);
                                        break;
                                    // DsDv
                                    case 4:
                                        cnt= mjd_transitionWASPOneThread(m, this->qv_basis, d, tol, mode,
                                                                    this->x_eps, this->x_eps,
                                                                    (int)(this->x_frac_wasp*m->nv),
                                                                    DataAt(this->CT,
                                                                           t * (dim_sensor * dim_state_derivative) + m->
                                                                           nv * dim_sensor), this->DsDv[t], mjDsDv);
                                        break;
                                    // DsDu
                                    default:
                                        cnt= mjd_transitionWASPOneThread(m, this->u_basis, d, tol, mode,
                                                                    this->u_eps, this->u_eps,
                                                                    (int)(this->u_frac_wasp*m->nu),
                                                                    DataAt(this->DT, t * (dim_sensor * dim_action)),
                                                                    this->DsDu[t], mjDsDu);
                                        break;
                                }
                            }
                            num_dynamics_called+=cnt;
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
