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
// Base class for all iterative LQ game solvers.
// Structured so that derived classes may only modify the `ModifyLQStrategies`
// and `HasConverged` virtual functions.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/player_cost.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/lq_solver.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/loop_timer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/quadratic_cost_approximation.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <chrono>
#include <memory>
#include <numeric>
#include <vector>

namespace ilqgames {

namespace {

// Multiply all alphas in a set of strategies by the given constant.
void ScaleAlphas(float scaling, std::vector<Strategy>* strategies) {
  CHECK_NOTNULL(strategies);

  for (auto& strategy : *strategies) {
    for (auto& alpha : strategy.alphas) alpha *= scaling;
  }
}

}  // anonymous namespace

std::shared_ptr<SolverLog> ILQSolver::Solve(bool* success, Time max_runtime) {
  const auto solver_call_time = Clock::now();

  // Create a new log.
  std::shared_ptr<SolverLog> log = CreateNewLog();

  // Last and current operating points. Make sure the last one starts from the
  // current state so that the current one will start there as well.
  // NOTE: setting the current operating point to start at x0 is critical to the
  // constraint satisfaction check at the first iteration.
  OperatingPoint last_operating_point(problem_->CurrentOperatingPoint());
  OperatingPoint current_operating_point(problem_->CurrentOperatingPoint());
  current_operating_point.xs[0] = problem_->InitialState();
  last_operating_point.xs[0] = problem_->InitialState();

  // Current strategies.
  std::vector<Strategy> current_strategies(problem_->CurrentStrategies());

  // Things to keep track of during each iteration.
  size_t num_iterations = 0;
  bool has_converged = false;

  // Swap operating points and compute new current operating point. Future
  // operating points will be computed during the call to `ModifyLQStrategies`
  // which occurs after solving the LQ game.
  std::vector<float> total_costs;
  last_operating_point.swap(current_operating_point);
  CurrentOperatingPoint(last_operating_point, current_strategies,
                        &current_operating_point);

  // Compute total costs.
  TotalCosts(current_operating_point, &total_costs);

  // Log current iterate.
  Time elapsed = 0.0;
  log->AddSolverIterate(current_operating_point, current_strategies,
                        total_costs, elapsed, has_converged);

  // Quadraticize costs before first iteration. Subsequent quadraticizations
  // will happen inside the linesearch every time we compute the merit function.
  ComputeCostQuadraticization(current_operating_point, &cost_quadraticization_);

  // Maintain delta_xs and costates for linesearch.
  std::vector<VectorXf> delta_xs;
  std::vector<std::vector<VectorXf>> costates;

  // Main loop with timer for anytime execution.
  while (num_iterations < params_.max_solver_iters && !has_converged &&
         elapsed < max_runtime - timer_.RuntimeUpperBound()) {
    // Start loop timer.
    timer_.Tic();

    // New iteration.
    num_iterations++;

    // Linearize dynamics about the new operating point, only if the system
    // can't be treated as linear from the outset, in which case we've already
    // linearized it.
    // NOTE: we are already computing a new quadraticization
    // during the linesearch process.
    if (!problem_->Dynamics()->TreatAsLinear())
      ComputeLinearization(current_operating_point, &linearization_);

    // Solve LQ game.
    current_strategies = lq_solver_->Solve(
        linearization_, cost_quadraticization_,
        problem_->InitialState() - current_operating_point.xs.front(),
        &delta_xs, &costates);

    // Modify this LQ solution.
    if (!ModifyLQStrategies(delta_xs, costates, &current_strategies,
                            &current_operating_point, &has_converged)) {
      // Maybe emit warning if exiting early.
      VLOG(1) << "Solver exited due to linesearch failure.";

      // Handle success flag.
      if (success) *success = false;

      return log;
    }

    // Compute total costs and check if we've converged.
    TotalCosts(current_operating_point, &total_costs);

    // Record loop runtime.
    elapsed += timer_.Toc();

    // Log current iterate.
    log->AddSolverIterate(current_operating_point, current_strategies,
                          total_costs, elapsed, has_converged);
  }

  // Handle success flag.
  if (success) *success = true;

  return log;
}

void ILQSolver::CurrentOperatingPoint(
    const OperatingPoint& last_operating_point,
    const std::vector<Strategy>& current_strategies,
    OperatingPoint* current_operating_point) const {
  CHECK_NOTNULL(current_operating_point);

  // Initialize time.
  current_operating_point->t0 = last_operating_point.t0;

  // Integrate dynamics and populate operating point, one time step at a time.
  VectorXf x(last_operating_point.xs[0]);
  for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
    const Time t = RelativeTimeTracker::RelativeTime(kk);

    // Unpack.
    const VectorXf delta_x = x - last_operating_point.xs[kk];
    const auto& last_us = last_operating_point.us[kk];
    auto& current_us = current_operating_point->us[kk];

    // Record state.
    current_operating_point->xs[kk] = x;

    // Compute and record control for each player.
    for (PlayerIndex jj = 0; jj < problem_->Dynamics()->NumPlayers(); jj++) {
      const auto& strategy = current_strategies[jj];
      current_us[jj] = strategy(kk, delta_x, last_us[jj]);
    }

    // Integrate dynamics for one time step.
    if (kk < time::kNumTimeSteps - 1)
      x = problem_->Dynamics()->Integrate(t, time::kTimeStep, x, current_us);
  }
}

// bool ILQSolver::HasConverged(const OperatingPoint& last_op,
//                              const OperatingPoint& current_op) const {
//   for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
//     const float delta_x_distance = StateDistance(
//         current_op.xs[kk], last_op.xs[kk], params_.trust_region_dimensions);

//     if (delta_x_distance > params_.convergence_tolerance) return false;
//   }

//   return true;
// }

void ILQSolver::TotalCosts(const OperatingPoint& current_op,
                           std::vector<float>* total_costs) const {
  // Initialize appropriately.
  if (total_costs->size() != problem_->PlayerCosts().size())
    total_costs->resize(problem_->PlayerCosts().size());
  for (PlayerIndex ii = 0; ii < problem_->PlayerCosts().size(); ii++) {
    if (problem_->PlayerCosts()[ii].IsTimeAdditive())
      (*total_costs)[ii] = 0.0;
    else if (problem_->PlayerCosts()[ii].IsMaxOverTime())
      (*total_costs)[ii] = -constants::kInfinity;
    else
      (*total_costs)[ii] = constants::kInfinity;
  }

  // Accumulate costs.
  for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
    const Time t = RelativeTimeTracker::RelativeTime(kk);

    for (size_t ii = 0; ii < problem_->PlayerCosts().size(); ii++) {
      const float current_cost = problem_->PlayerCosts()[ii].Evaluate(
          t, current_op.xs[kk], current_op.us[kk]);

      if (problem_->PlayerCosts()[ii].IsTimeAdditive())
        (*total_costs)[ii] += problem_->PlayerCosts()[ii].Evaluate(
            t, current_op.xs[kk], current_op.us[kk]);
      else if (problem_->PlayerCosts()[ii].IsMaxOverTime() &&
               current_cost > (*total_costs)[ii]) {
        (*total_costs)[ii] = current_cost;
        problem_->PlayerCosts()[ii].SetTimeOfExtremeCost(kk);
      } else if (problem_->PlayerCosts()[ii].IsMinOverTime()) {
        if (current_cost < (*total_costs)[ii]) {
          (*total_costs)[ii] = current_cost;
          problem_->PlayerCosts()[ii].SetTimeOfExtremeCost(kk);
        }
      }
    }
  }
}

float ILQSolver::StateDistance(const VectorXf& x1, const VectorXf& x2,
                               const std::vector<Dimension>& dims) const {
  auto total_distance = [&dims](const VectorXf& x1, const VectorXf& x2) {
    if (dims.empty()) return (x1 - x2).cwiseAbs().maxCoeff();

    float distance = 0.0;
    for (const Dimension dim : dims) distance += std::abs(x1(dim) - x2(dim));

    return distance;
  };  // total_distance

  if (problem_->Dynamics()->TreatAsLinear()) {
    const auto& dyn = problem_->FlatDynamics();

    // If singular return infinite distance and throw a warning. Otherwise, use
    // base class implementation but for nonlinear system states.
    if (dyn.IsLinearSystemStateSingular(x1) ||
        dyn.IsLinearSystemStateSingular(x2)) {
      LOG(WARNING)
          << "Singular state encountered when computing state distance.";
      return std::numeric_limits<float>::infinity();
    }

    return total_distance(dyn.FromLinearSystemState(x1),
                          dyn.FromLinearSystemState(x2));
  }

  return total_distance(x1, x2);
}

bool ILQSolver::ModifyLQStrategies(
    const std::vector<VectorXf>& delta_xs,
    const std::vector<std::vector<VectorXf>>& costates,
    std::vector<Strategy>* strategies, OperatingPoint* current_operating_point,
    bool* has_converged) {
  CHECK_NOTNULL(strategies);
  CHECK_NOTNULL(current_operating_point);
  CHECK_NOTNULL(has_converged);

  // DEBUG: show how alphas are decaying - i.e., we're finding a fixed point.
  //  std::cout << strategies->front().alphas.front().squaredNorm() <<
  //  std::endl;

  // Precompute expected decrease before we do anything else.
  expected_decrease_ = ExpectedDecrease(*strategies, delta_xs, costates);
  //  std::cout << expected_decrease_ << std::endl;

  // Every computation of the merit function will overwrite the current cost
  // quadraticization, so first swap it with the previous one so we retain a
  // copy.
  if (params_.linesearch) last_cost_quadraticization_.swap(cost_quadraticization_);

  // Initially scale alphas by a fixed amount to avoid unnecessary
  // backtracking.
  // NOTE: use adaptive initialization here based on history.
  ScaleAlphas(params_.initial_alpha_scaling, strategies);

  // Compute next operating point and keep track of whether it satisfies the
  // Armijo condition.
  const OperatingPoint last_operating_point(*current_operating_point);
  float current_stepsize = params_.initial_alpha_scaling;
  CurrentOperatingPoint(last_operating_point, *strategies,
                        current_operating_point);
  if (!params_.linesearch) return true;

  // Keep reducing alphas until we satisfy the Armijo condition.
  for (size_t ii = 0; ii < params_.max_backtracking_steps; ii++) {
    // Compute merit function value.
    const float current_merit_function_value =
        MeritFunction(*current_operating_point, costates);

    // Check Armijo condition.
    if (CheckArmijoCondition(current_merit_function_value, current_stepsize)) {
      // Success! Update cached terms and check convergence.
      *has_converged = HasConverged(current_merit_function_value);
      last_merit_function_value_ = current_merit_function_value;
      return true;
    }

    // Scale down the alphas and try again.
    ScaleAlphas(params_.geometric_alpha_scaling, strategies);
    current_stepsize *= params_.geometric_alpha_scaling;
    CurrentOperatingPoint(last_operating_point, *strategies,
                          current_operating_point);
  }

  // Output a warning. Solver should revert to last valid operating point.
  VLOG(1) << "Exceeded maximum number of backtracking steps.";
  return false;
}

bool ILQSolver::CheckArmijoCondition(float current_merit_function_value,
                                     float current_stepsize) const {
  // Adjust total expected decrease.
  const float scaled_expected_decrease = params_.expected_decrease_fraction *
                                         current_stepsize * expected_decrease_;

  // std::cout << "Improvement = "
  //           << last_merit_function_value_ - current_merit_function_value
  //           << ", expected = " << scaled_expected_decrease << std::endl;

  return (last_merit_function_value_ - current_merit_function_value >=
          scaled_expected_decrease);
}

float ILQSolver::ExpectedDecrease(
    const std::vector<Strategy>& strategies,
    const std::vector<VectorXf>& delta_xs,
    const std::vector<std::vector<VectorXf>>& costates) const {
  float expected_decrease = 0.0;
  for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
    const auto& lin = linearization_[kk];

    // Separate x expected decrease per step at each time (saves computation).
    const size_t xdim = problem_->Dynamics()->XDim();
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const auto& quad = cost_quadraticization_[kk][ii];
      const auto& costate = costates[kk][ii];
      const auto& neg_ui =
          strategies[ii].alphas[kk];  // NOTE: could also evaluate delta u on
                                      // delta x to be more precise.

      // Accumulate state and control contributions.
      // NOTE: ignoring costate contributions because right now, these are actually
      //       only changes in costate, so the values here are not quite right.
      const VectorXf dLdu =
          quad.control.at(ii).grad; // - lin.Bs[ii].transpose() * costate;
      expected_decrease -= neg_ui.transpose() * quad.control.at(ii).hess * dLdu;

      if (kk > 0) {
        const auto& last_costate = costates[kk - 1][ii];
        const VectorXf dLdx =
            quad.state.grad; // - lin.A.transpose() * costate + last_costate;
        expected_decrease -= delta_xs[kk].transpose() * quad.state.hess * dLdx;
      }
    }
  }

  return expected_decrease;
}

float ILQSolver::MeritFunction(
    const OperatingPoint& current_op,
    const std::vector<std::vector<VectorXf>>& costates) {
  // First, quadraticize cost and linearize dynamics around this operating
  // point.
  ComputeCostQuadraticization(current_op, &cost_quadraticization_);

  // Accumulate gradients of Lagrangian (dynamics automatically satisfied).
  float merit = 0.0;
  for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
    const auto& lin = linearization_[kk];

    // Separate x expected decrease per step at each time (saves computation).
    const size_t xdim = problem_->Dynamics()->XDim();
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const auto& quad = cost_quadraticization_[kk][ii];
      const auto& costate = costates[kk][ii];

      // Accumulate state and control contributions.
      // NOTE: ignoring costate contributions because right now, these are actually
      //       only changes in costate, so the values here are not quite right.
      const VectorXf dLdu =
          quad.control.at(ii).grad; // - lin.Bs[ii].transpose() * costate;
      merit += dLdu.squaredNorm();

      if (kk > 0) {
        const auto& last_costate = costates[kk - 1][ii];
        const VectorXf dLdx =
            quad.state.grad; // - lin.A.transpose() * costate + last_costate;
        merit += dLdx.squaredNorm();
      }
    }
  }

  return 0.5 * merit;
}

void ILQSolver::ComputeLinearization(
    const OperatingPoint& op,
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Check if linearization is the right length.
  if (linearization->size() != op.xs.size())
    linearization->resize(op.xs.size());

  // Cast dynamics to appropriate type.
  const auto dyn = static_cast<const MultiPlayerDynamicalSystem*>(
      problem_->Dynamics().get());

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < op.xs.size(); kk++) {
    const Time t = RelativeTimeTracker::RelativeTime(kk);
    (*linearization)[kk] = dyn->Linearize(t, op.xs[kk], op.us[kk]);
  }
}

void ILQSolver::ComputeLinearization(
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Cast dynamics to appropriate type and make sure the system is
  // linearizable.
  CHECK(problem_->Dynamics()->TreatAsLinear());
  const auto& dyn = problem_->FlatDynamics();

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < linearization->size(); kk++)
    (*linearization)[kk] = dyn.LinearizedSystem();
}

void ILQSolver::ComputeCostQuadraticization(
    const OperatingPoint& op,
    std::vector<std::vector<QuadraticCostApproximation>>* q) {
  for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
    const Time t = RelativeTimeTracker::RelativeTime(kk);
    const auto& x = op.xs[kk];
    const auto& us = op.us[kk];

    // Quadraticize costs.
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const PlayerCost& cost = problem_->PlayerCosts()[ii];

      if (cost.IsTimeAdditive() ||
          problem_->PlayerCosts()[ii].TimeOfExtremeCost() == kk)
        (*q)[kk][ii] = cost.Quadraticize(t, x, us);
      else
        (*q)[kk][ii] = cost.QuadraticizeControlCosts(t, x, us);
    }
  }
}

}  // namespace ilqgames
