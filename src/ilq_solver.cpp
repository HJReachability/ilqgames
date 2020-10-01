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
  const auto solver_call_time = clock::now();

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

  // Reset all constraint barrier weights to unity.
  for (auto& cost : problem_->PlayerCosts()) cost.ResetBarrierWeights();

  // Things to keep track of during each iteration.
  size_t num_iterations = 0;
  size_t num_iterations_since_barrier_rescaling = 0;
  bool has_converged = false;

  // Turn barriers on/off.
  auto turn_barriers_on = [this]() {
    for (auto& cost : problem_->PlayerCosts()) cost.TurnBarriersOn();
  };  // turn_barriers_on

  auto turn_barriers_off = [this]() {
    for (auto& cost : problem_->PlayerCosts()) cost.TurnBarriersOff();
  };  // turn_barriers_off

  // Swap operating points and compute new current operating point. Future
  // operating points will be computed during the call to `ModifyLQStrategies`
  // which occurs after solving the LQ game.
  bool was_operating_point_feasible;
  std::vector<float> total_costs;
  last_operating_point.swap(current_operating_point);
  CurrentOperatingPoint(last_operating_point, current_strategies,
                        &current_operating_point,
                        &was_operating_point_feasible);

  // Compute total costs.
  TotalCosts(current_operating_point, &total_costs);

  // Log current iterate.
  Time elapsed = 0.0;
  log->AddSolverIterate(current_operating_point, current_strategies,
                        total_costs, elapsed, has_converged);

  // Main loop with timer for anytime execution.
  while (num_iterations < params_.max_solver_iters && !has_converged &&
         elapsed < max_runtime - timer_.RuntimeUpperBound()) {
    // Start loop timer.
    timer_.Tic();

    // New iteration.
    num_iterations++;
    num_iterations_since_barrier_rescaling++;

    // Maybe rescale barrier barrier weights.
    if (num_iterations_since_barrier_rescaling >
        params_.barrier_scaling_iters) {
      num_iterations_since_barrier_rescaling = 0;
      for (PlayerCost& cost : problem_->PlayerCosts())
        cost.ScaleBarrierWeights(params_.geometric_barrier_scaling);
    }

    // If operating point is feasible, turn on barriers. If it is
    // not feasible, then turn them off.
    if (was_operating_point_feasible)
      turn_barriers_on();
    else
      turn_barriers_off();

    // Linearize dynamics about the new operating point, only if the system
    // can't be treated as linear from the outset, in which case we've already
    // linearized it.
    // NOTE: we are already computing a new quadraticization
    // during the linesearch process.
    if (!problem_->Dynamics()->TreatAsLinear())
      ComputeLinearization(current_operating_point, &linearization_);

    // Do quadraticize in the first iteration.
    if (num_iterations == 1)
      ComputeCostQuadraticization(current_operating_point,
                                  &cost_quadraticization_);

    // Solve LQ game.
    current_strategies = lq_solver_->Solve(
        linearization_, cost_quadraticization_, problem_->InitialState());

    // Modify this LQ solution.
    if (!ModifyLQStrategies(&current_strategies, &current_operating_point,
                            &was_operating_point_feasible)) {
      // Maybe emit warning if exiting early.
      if (num_iterations == 1) {
        VLOG(1)
            << "Solver exited after during first iteration, which may indicate "
               "an infeasible initial operating point.";

        if (was_operating_point_feasible)
          VLOG(1) << "Previous operating point was feasible.";
        else {
          VLOG(1) << "Previous operating point was infeasible.";
        }
      }

      // Handle success flag.
      if (success) *success = false;

      return log;
    }

    // Compute total costs and check if we've converged.
    TotalCosts(current_operating_point, &total_costs);
    has_converged = HasConverged(last_operating_point, current_operating_point);

    // Record loop runtime.
    elapsed = timer_.Toc();

    // Log current iterate.
    log->AddSolverIterate(current_operating_point, current_strategies,
                          total_costs, elapsed, has_converged);
  }

  CHECK(!problem_->PlayerCosts().front().AreBarriersOn() ||
        was_operating_point_feasible);

  // Maybe emit warning if exiting early.
  if (num_iterations == 1) {
    VLOG(1) << "Solver exited after only 1 iteration but passed "
               "backtracking checks, which may indicate an almost "
               "converged initial operating point and strategies.";
  }

  if (!was_operating_point_feasible) {
    VLOG(1) << "Solver found an infeasible solution. Failing.";

    // Handle success flag.
    if (success) *success = false;

    return log;
  }

  // Handle success flag.
  if (success) *success = true;

  // Update problem solution by convention.
  problem_->OverwriteSolution(current_operating_point, current_strategies);

  return log;
}

void ILQSolver::CurrentOperatingPoint(
    const OperatingPoint& last_operating_point,
    const std::vector<Strategy>& current_strategies,
    OperatingPoint* current_operating_point, bool* satisfies_barriers) const {
  CHECK_NOTNULL(current_operating_point);

  // Initialize time, convergence, and barrier satisfaction checks.
  current_operating_point->t0 = last_operating_point.t0;
  if (satisfies_barriers) *satisfies_barriers = true;

  // Integrate dynamics and populate operating point, one time step at a time.
  VectorXf x(last_operating_point.xs[0]);
  for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);

    // Unpack.
    const VectorXf delta_x = x - last_operating_point.xs[kk];
    const auto& last_us = last_operating_point.us[kk];
    auto& current_us = current_operating_point->us[kk];

    // Check barriers.
    auto check_all_barriers = [this](Time t, const VectorXf& x,
                                     const std::vector<VectorXf>& us) {
      for (const auto& cost : problem_->PlayerCosts()) {
        if (!cost.CheckBarriers(t, x, us)) return false;
      }
      return true;
    };  // check_all_barriers

    const bool checked_barriers = check_all_barriers(t, x, current_us);

    if (satisfies_barriers) *satisfies_barriers &= checked_barriers;

    // Record state.
    current_operating_point->xs[kk] = x;

    // Compute and record control for each player.
    for (PlayerIndex jj = 0; jj < problem_->Dynamics()->NumPlayers(); jj++) {
      const auto& strategy = current_strategies[jj];
      current_us[jj] = strategy(kk, delta_x, last_us[jj]);
    }

    // Integrate dynamics for one time step.
    if (kk < problem_->NumTimeSteps() - 1)
      x = problem_->Dynamics()->Integrate(t, problem_->TimeStep(), x,
                                          current_us);
  }
}

// bool ILQSolver::HasConverged(const OperatingPoint& last_op,
//                              const OperatingPoint& current_op) const {
//   for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
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
  for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);

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

bool ILQSolver::ModifyLQStrategies(std::vector<Strategy>* strategies,
                                   OperatingPoint* current_operating_point,
                                   bool* is_new_operating_point_feasible) {
  CHECK_NOTNULL(strategies);
  CHECK_NOTNULL(current_operating_point);

  // DEBUG: show how alphas are decaying - i.e., we're finding a fixed point.
  //  std::cout << strategies->front().alphas.front().squaredNorm() <<
  //  std::endl;

  // Initially scale alphas by a fixed amount to avoid unnecessary
  // backtracking.
  ScaleAlphas(params_.initial_alpha_scaling, strategies);

  // Compute next operating point and keep track of whether it satisfies the
  // Armijo condition.
  const OperatingPoint last_operating_point(*current_operating_point);
  float current_stepsize = params_.initial_alpha_scaling;
  float current_kkt_squared_error = constants::kInfinity;
  CurrentOperatingPoint(last_operating_point, *strategies,
                        current_operating_point,
                        is_new_operating_point_feasible);

  if (!params_.linesearch) return true;

  // Keep reducing alphas until we satisfy the Armijo condition.
  for (size_t ii = 0; ii < params_.max_backtracking_steps; ii++) {
    if (CheckArmijoCondition(*current_operating_point, current_stepsize,
                             &current_kkt_squared_error)) {
      // Success! Update cached terms.
      last_kkt_squared_error_ = current_kkt_squared_error;
      expected_decrease_.release();
      return true;
    }

    ScaleAlphas(params_.geometric_alpha_scaling, strategies);
    current_stepsize *= params_.geometric_alpha_scaling;
    CurrentOperatingPoint(last_operating_point, *strategies,
                          current_operating_point,
                          is_new_operating_point_feasible);
  }

  // Output a warning. Solver should revert to last valid operating point.
  VLOG(1) << "Exceeded maximum number of backtracking steps.";
  return false;
}

bool ILQSolver::CheckArmijoCondition(const OperatingPoint& current_op,
                                     float current_stepsize,
                                     float* current_kkt_squared_error) {
  CHECK_NOTNULL(current_kkt_squared_error);

  // Compute current KKT squared error. In the process, this will compute a new
  // quadratic approximation at the current operating point and save the old
  // quadraticization.
  // NOTE: Currently, all KKT computations assume an *open loop* KKT structure.
  // It remains to show that this is correct for *feedback* games, but
  // empirically it seems to work ok.
  *current_kkt_squared_error = KKTSquaredError(current_op);

  // Compute total expected decrease of KKT squared error if not already
  // computed.
  if (!expected_decrease_.get()) {
    expected_decrease_ = make_unique<float>(0.0);
    for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
      for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
        const auto& quad = cost_quadraticization_[kk][ii];

        const Eigen::HouseholderQR<MatrixXf> state_qr(quad.state.hess);
        const float current_expected_decrease =
            state_qr.solve(quad.state.grad).squaredNorm();
        *expected_decrease_ += std::accumulate(
            quad.control.begin(), quad.control.end(), current_expected_decrease,
            [](float total,
               const std::pair<PlayerIndex, SingleCostApproximation>& entry) {
              const Eigen::HouseholderQR<MatrixXf> control_qr(
                  entry.second.hess);
              return total + control_qr.solve(entry.second.grad).squaredNorm();
            });
      }
    }
  }
  // Adjust total expected decrease.
  const float scaled_expected_decrease = *expected_decrease_ * 2.0 *
                                         current_stepsize *
                                         params_.expected_decrease_fraction;

  // std::cout << "expected: " << total_expected_decrease << "\n"
  //           << "actual: "
  //           << last_kkt_squared_error_ - *current_kkt_squared_error
  //           << std::endl;

  return (last_kkt_squared_error_ - *current_kkt_squared_error >=
          scaled_expected_decrease);
}

float ILQSolver::KKTSquaredError(const OperatingPoint& current_op) {
  // NOTE: Currently, all KKT computations assume an *open loop* KKT structure.
  // It remains to show that this is correct for *feedback* games, but
  // empirically it seems to work ok.
  // NOTE: this will update the current quadraticization and save the old one.
  last_cost_quadraticization_.swap(cost_quadraticization_);
  ComputeCostQuadraticization(current_op, &cost_quadraticization_);

  float total_squared_error = 0.0;
  for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const auto& quad = cost_quadraticization_[kk][ii];

      // Accumulate state and control gradient squared norms.
      const float current_squared_error = quad.state.grad.squaredNorm();
      total_squared_error += std::accumulate(
          quad.control.begin(), quad.control.end(), current_squared_error,
          [](float total,
             const std::pair<PlayerIndex, SingleCostApproximation>& entry) {
            return total + entry.second.grad.squaredNorm();
          });
    }
  }

  return total_squared_error;
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
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);
    (*linearization)[kk] = dyn->Linearize(t, op.xs[kk], op.us[kk]);
  }
}

void ILQSolver::ComputeLinearization(
    std::vector<LinearDynamicsApproximation>* linearization) {
  CHECK_NOTNULL(linearization);

  // Cast dynamics to appropriate type and make sure the system is linearizable.
  CHECK(problem_->Dynamics()->TreatAsLinear());
  const auto& dyn = problem_->FlatDynamics();

  // Populate one timestep at a time.
  for (size_t kk = 0; kk < linearization->size(); kk++)
    (*linearization)[kk] = dyn.LinearizedSystem();
}

void ILQSolver::ComputeCostQuadraticization(
    const OperatingPoint& op,
    std::vector<std::vector<QuadraticCostApproximation>>* q) {
  for (size_t kk = 0; kk < problem_->NumTimeSteps(); kk++) {
    const Time t =
        problem_->InitialTime() + problem_->ComputeRelativeTimeStamp(kk);
    const auto& x = op.xs[kk];
    const auto& us = op.us[kk];

    // Quadraticize costs.
    for (PlayerIndex ii = 0; ii < problem_->Dynamics()->NumPlayers(); ii++) {
      const PlayerCost& cost = problem_->PlayerCosts()[ii];

      if (cost.IsTimeAdditive() ||
          problem_->PlayerCosts()[ii].TimeOfExtremeCost() == kk)
        (*q)[kk][ii] = cost.Quadraticize(t, kk, x, us);
      else
        (*q)[kk][ii] = cost.QuadraticizeBarriersAndControlCosts(t, x, us);
    }
  }
}

}  // namespace ilqgames
