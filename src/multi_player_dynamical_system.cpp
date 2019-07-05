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
// Base class for all multi-player dynamical systems. Supports (discrete-time)
// linearization and integration.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/multi_player_dynamical_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

VectorXf MultiPlayerDynamicalSystem::Integrate(
    Time t0, Time time_step, const VectorXf& x0,
    const std::vector<VectorXf>& us) const {
  // Number of integration steps and corresponding time step.
  constexpr size_t kNumIntegrationSteps = 2;
  const double dt = time_step / static_cast<Time>(kNumIntegrationSteps);

  // RK4 integration. See https://en.wikipedia.org/wiki/Runge-Kutta_methods for
  // further details.
  VectorXf x(x0);
  for (Time t = t0; t < t0 + time_step; t += dt) {
    const VectorXf k1 = dt * Evaluate(t, x, us);
    const VectorXf k2 = dt * Evaluate(t + 0.5 * dt, x + 0.5 * k1, us);
    const VectorXf k3 = dt * Evaluate(t + 0.5 * dt, x + 0.5 * k2, us);
    const VectorXf k4 = dt * Evaluate(t + dt, x + k3, us);

    x += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
  }

  return x;
}

VectorXf MultiPlayerDynamicalSystem::Integrate(
    Time t0, Time time_step, Time t, const VectorXf& x0,
    const OperatingPoint& operating_point,
    const std::vector<Strategy>& strategies) const {
  CHECK_GE(t, t0);
  CHECK_GE(t0, operating_point.t0);
  CHECK_EQ(strategies.size(), NumPlayers());

  std::vector<VectorXf> us(NumPlayers());

  // Compute current timestep.
  const Time relative_t0 = t0 - operating_point.t0;
  const size_t current_timestep = static_cast<size_t>(relative_t0 / time_step);

  // Handle case where 't0' is after 'operating_point.t0' by integrating from
  // 't0' to the next discrete timestep.
  VectorXf x(x0);
  if (t0 > operating_point.t0) {
    const Time remaining_time_this_step =
        time_step * (current_timestep + 1) - relative_t0;

    // Interpolate x0_ref.
    CHECK_LT(current_timestep + 1, operating_point.xs.size());
    const float frac = remaining_time_this_step / time_step;
    const VectorXf x0_ref =
        frac * operating_point.xs[current_timestep] +
        (1.0 - frac) * operating_point.xs[current_timestep + 1];

    // Populate controls for each player.
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
      us[ii] = strategies[ii](current_timestep, x0 - x0_ref,
                              operating_point.us[current_timestep][ii]);

    x = Integrate(t0, remaining_time_this_step, x0, us);
  }

  // Integrate forward step by step up to timestep including t.
  const Time relative_t = t - operating_point.t0;
  const size_t final_timestep = static_cast<size_t>(relative_t / time_step);
  const Time remaining_time_final_step =
      relative_t -
      time_step * final_timestep;  // NB: opposite direction as above.

  for (size_t kk = current_timestep + 1; kk < final_timestep; kk++) {
    const Time time = operating_point.t0 + kk * time_step;

    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++)
      us[ii] = strategies[ii](kk, x - operating_point.xs[kk],
                              operating_point.us[kk][ii]);

    x = Integrate(time, time_step, x, us);
  }

  // Integrate forward from this timestep to t.
  for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) {
    us[ii] =
        strategies[ii](final_timestep, x - operating_point.xs[final_timestep],
                       operating_point.us[final_timestep][ii]);
  }

  return Integrate(operating_point.t0 + time_step * final_timestep,
                   remaining_time_final_step, x, us);
}

}  // namespace ilqgames
