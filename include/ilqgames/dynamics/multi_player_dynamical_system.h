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

#ifndef ILQGAMES_DYNAMICS_MULTI_PLAYER_DYNAMICAL_SYSTEM_H
#define ILQGAMES_DYNAMICS_MULTI_PLAYER_DYNAMICAL_SYSTEM_H

#include <ilqgames/dynamics/multi_player_integrable_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

class MultiPlayerDynamicalSystem : public MultiPlayerIntegrableSystem {
 public:
  virtual ~MultiPlayerDynamicalSystem() {}

  // Compute time derivative of state.
  virtual VectorXf Evaluate(Time t, const VectorXf& x,
                            const std::vector<VectorXf>& us) const = 0;

  // Compute a discrete-time Jacobian linearization.
  virtual LinearDynamicsApproximation Linearize(
      Time t, const VectorXf& x, const std::vector<VectorXf>& us) const = 0;

  // Integrate these dynamics forward in time.
  VectorXf Integrate(Time t0, Time time_interval, const VectorXf& x0,
                     const std::vector<VectorXf>& us) const;

  // Getters.
  virtual Dimension UDim(PlayerIndex player_idx) const = 0;
  virtual PlayerIndex NumPlayers() const = 0;

 protected:
  explicit MultiPlayerDynamicalSystem(Dimension xdim)
      : MultiPlayerIntegrableSystem(xdim) {}

};  //\class MultiPlayerDynamicalSystem

}  // namespace ilqgames

#endif
