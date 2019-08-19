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
// Base class for all multi-player flat systems. Supports (discrete-time)
// linearization and integration.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

VectorXf MultiPlayerFlatSystem::Integrate(
    Time time_interval, const VectorXf& xi0,
    const std::vector<VectorXf>& vs) const {
  // Number of integration steps and corresponding time step.
  constexpr size_t kNumIntegrationSteps = 2;
  const double dt = time_step_ / static_cast<Time>(kNumIntegrationSteps);

  CHECK_NOTNULL(continuous_linear_system_.get());
  auto xi_dot = [this, &vs](const VectorXf& xi) {
    VectorXf deriv = this->continuous_linear_system_->A * xi;
    for (size_t ii=0; ii < NumPlayers(); ii++)
      deriv += this->continuous_linear_system_->Bs[ii] * vs[ii];
    
    return deriv;
  }; // xi_dot

  // RK4 integration. See https://en.wikipedia.org/wiki/Runge-Kutta_methods for
  // further details.
  VectorXf xi(xi0);
  for (Time t = 0.0; t < time_interval - 0.5*dt; t += dt) {
    const VectorXf k1 = dt * xi_dot(xi);
    const VectorXf k2 = dt * xi_dot(xi + 0.5 * k1);
    const VectorXf k3 = dt * xi_dot(xi + 0.5 * k2);
    const VectorXf k4 = dt * xi_dot(xi + k3);

    xi += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
  }

  return xi;
}

VectorXf MultiPlayerFlatSystem::Integrate(
    Time t0, Time t, const VectorXf& xi0,
    const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  CHECK_GE(t, t0);
  CHECK_GE(t0, operating_point.t0);
  CHECK_EQ(strategies.size(), NumPlayers());

  std::vector<VectorXf> vs(NumPlayers());

  // Handle case where 't0' is after 'operating_point.t0' by integrating from
  // 't0' to the next discrete timestep.
  VectorXf xi(xi0);
  if (t0 > operating_point.t0)
    xi = IntegrateToNextTimeStep(t0, xi0, operating_point, strategies);

  // Compute current timestep and final timestep.
  const Time relative_t0 = t0 - operating_point.t0;
  const size_t current_timestep = static_cast<size_t>(relative_t0 / time_step_);

  const Time relative_t = t - operating_point.t0;
  const size_t final_timestep = static_cast<size_t>(relative_t / time_step_);

  // Integrate forward step by step up to timestep including t.
  xi = Integrate(current_timestep + 1, final_timestep, xi,
                operating_point, strategies);

  // Integrate forward from this timestep to t.
  return IntegrateFromPriorTimeStep(t, xi, operating_point,
                                    strategies);
}

VectorXf MultiPlayerFlatSystem::Integrate(
    size_t initial_timestep, size_t final_timestep,
    const VectorXf& xi0, const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  VectorXf xi(xi0);
  std::vector<VectorXf> vs(NumPlayers());
  for (size_t kk = initial_timestep; kk < final_timestep; kk++) {
    const Time t = operating_point.t0 + kk * time_step_;

    // Populate controls for all players.
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
      vs[ii] = strategies[ii](kk, xi - operating_point.xs[kk],
                              operating_point.us[kk][ii]);

    xi = Integrate(t, xi, vs);
  }

  return xi;
}

VectorXf MultiPlayerFlatSystem::IntegrateToNextTimeStep(
    Time t0, const VectorXf& xi0,
    const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  CHECK_GE(t0, operating_point.t0);

  // Compute remaining time this timestep.
  const Time relative_t0 = t0 - operating_point.t0;
  const size_t current_timestep = static_cast<size_t>(relative_t0 / time_step_);
  const Time remaining_time_this_step =
      time_step_ * (current_timestep + 1) - relative_t0;

  // Interpolate x0_ref.
  CHECK_LT(current_timestep + 1, operating_point.xs.size());
  const float frac = remaining_time_this_step / time_step_;
  const VectorXf xi0_ref =
      frac * operating_point.xs[current_timestep] +
      (1.0 - frac) * operating_point.xs[current_timestep + 1];

  // Populate controls for each player.
  std::vector<VectorXf> vs(NumPlayers());
  for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
    vs[ii] = strategies[ii](current_timestep, xi0 - xi0_ref,
                            operating_point.us[current_timestep][ii]);

  return Integrate(remaining_time_this_step, xi0, vs);
}

VectorXf MultiPlayerFlatSystem::IntegrateFromPriorTimeStep(
    Time t, const VectorXf& xi0,
    const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  // Compute time until next timestep.
  const Time relative_t = t - operating_point.t0;
  const size_t current_timestep = static_cast<size_t>(relative_t / time_step_);
  const Time remaining_time_until_t = relative_t - time_step_ * current_timestep;

  // Populate controls for each player.
  std::vector<VectorXf> vs(NumPlayers());
  for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) {
    vs[ii] = strategies[ii](current_timestep,
                            xi0 - operating_point.xs[current_timestep],
                            operating_point.us[current_timestep][ii]);
  }

  return Integrate(remaining_time_until_t, xi0, vs);
}

}  // namespace ilqgames
