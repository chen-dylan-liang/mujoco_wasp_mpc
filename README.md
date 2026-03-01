# mujoco_wasp_mpc

**This repository is built upon the [mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc) codebase. We thank the authors for open-sourcing their work.**

## Overview

This repository extends the original `mujoco_mpc` implementation with a WASP-based model-derivative engine for gradient-based MPC.

The core contribution is `ModelDerivativesWASP` (`mjpc/planners/model_derivatives_wasp.h` and `mjpc/planners/model_derivatives_wasp.cc`), which replaces finite-difference (FD) model derivative evaluation with WASP-based differentiation. The implementation is designed to preserve the practical behavior and interfaces of the original FD pipeline while significantly improving derivative evaluation efficiency in MPC loops.

## Key Integration Idea

In upstream `mujoco_mpc`, gradient-based planners consume a `ModelDerivatives` interface, and FD derivatives are computed through MuJoCo simulator-level routines.

This project keeps that architecture and introduces a drop-in WASP implementation:

- `ModelDerivativesWASP` inherits from `ModelDerivatives`.
- Existing planner logic continues to consume a `ModelDerivatives*` pointer.
- The derivative backend can be switched between FD and WASP without changing planner algorithms.

This allows WASP acceleration to be integrated in an elegant and low-intrusion way, while maintaining compatibility with existing planner infrastructure.

## Core Class: `ModelDerivativesWASP`

`ModelDerivativesWASP` is the central class of this repository. It provides:

- FD-compatible lifecycle behavior (`Allocate`, `Reset`, `Compute`) so existing planner usage patterns remain valid.
- Flexible reset semantics for iterative MPC use.
- Dedicated management of additional WASP cache/basis memory required for fast repeated derivative calls.
- Parallel derivative computation across horizon time steps via the existing thread-pool execution model.
- Optional cache rollout/reuse behavior to reduce repeated work between MPC iterations.

## Respecting Simulator-Level Isolation

As in `mujoco_mpc` (where FD derivative internals are isolated in MuJoCo), this repository keeps WASP low-level differentiation internals outside planner code.

All MuJoCo-level WASP primitives (e.g., functions starting with `mj` such as `mj_zeroWASPCache`, `mjd_transitionWASP`, and related routines) are implemented in a separate repository:

- `wasp_differentiated_mujoco`

This preserves clean software boundaries:

- Planner/reasoning layer in this repo.
- Low-level WASP-enabled simulator differentiation layer in the MuJoCo fork/repo.

## Planner Integration (GD and iLQG)

After implementing `ModelDerivativesWASP`, the backend is integrated into both major gradient-based planners:

- Gradient Descent (GD): `mjpc/planners/gradient/planner.cc`
- iLQG: `mjpc/planners/ilqg/planner.cc`

Both planners allocate and maintain:

- an FD derivative engine (`fd_md`)
- a WASP derivative engine (`wasp_md`)
- a shared pointer (`model_derivative`) used by optimization code

At runtime, users can switch derivative engines through GUI controls (`MD Engine: FD/WASP`) without modifying task code. WASP tuning parameters are also exposed in GUI (e.g., WASP fractions/tolerances), enabling practical and transparent comparison between FD and WASP during experiments.

## Why This Engineering Is Non-Trivial

The integration challenge is not only replacing one derivative routine with another. It requires:

- preserving planner-level interface contracts and numerical behavior expectations
- introducing new cache/basis memory lifecycles without destabilizing resets and rollouts
- keeping parallel execution safe and efficient across horizon steps
- maintaining clean separation from simulator internals

This repository focuses on that systems-level integration quality: high performance, minimal algorithm-level disruption, and architectural cleanliness.
