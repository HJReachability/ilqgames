/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Parameters for solvers.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_SOLVER_SOLVER_PARAMS_H
#define ILQGAMES_SOLVER_SOLVER_PARAMS_H

#include <ilqgames/utils/types.h>

namespace ilqgames {

struct SolverParams {
  // Consider a solution converged once max elementwise difference is below this
  // tolerance or solver has exceeded a maximum number of iterations.
  float convergence_tolerance = 1e-1;
  size_t max_solver_iters = 1000;

  // Linesearch parameters. If flag is set 'true', then applied initial alpha
  // scaling to all strategies and backs off geometrically at the given rate for
  // the specified number of steps.
  bool linesearch = true;
  float initial_alpha_scaling = 0.5;
  float geometric_alpha_scaling = 0.5;
  size_t max_backtracking_steps = 10;
  float expected_decrease_fraction = 0.1;

  // Whether solver should shoot for an open loop or feedback Nash.
  bool open_loop = false;

  // State and control regularization.
  float state_regularization = 0.0;
  float control_regularization = 0.0;

  // Augmented Lagrangian parameters.
  size_t unconstrained_solver_max_iters = 10;
  float geometric_mu_scaling = 1.1;
  float geometric_mu_downscaling = 0.5;
  float geometric_lambda_downscaling = 0.5;
  float constraint_error_tolerance = 1e-1;

  // Should the solver reset problem/constraint params to their initial values.
  // NOTE: defaults to true.
  bool reset_problem = true;
  bool reset_lambdas = true;
  bool reset_mu = true;
};  // struct SolverParams

}  // namespace ilqgames

#endif
