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

#ifndef ILQGAMES_DYNAMICS_MULTI_PLAYER_FLAT_SYSTEM_H
#define ILQGAMES_DYNAMICS_MULTI_PLAYER_FLAT_SYSTEM_H

#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/strategy.h>
#include <ilqgames/utils/types.h>

#include <vector>

namespace ilqgames {

class MultiPlayerFlatSystem {
 public:
  virtual ~MultiPlayerFlatSystem() {}

  // Compute time derivative of state.
  virtual VectorXf Evaluate(const VectorXf& x,
                            const std::vector<VectorXf>& us) const = 0;

  // Utilities for feedback linearization.
  virtual MatrixXf InverseDecouplingMatrix(const VectorXf& x) const = 0;

  virtual VectorXf AffineTerm(const VectorXf& x) const = 0;

  virtual VectorXf LinearizingControl(const VectorXf& x, 
                                      const VectorXf& v) const = 0;

  virtual VectorXf ToLinearSystemState(const VectorXf& x) const = 0;

  virtual VectorXf FromLinearSystemState(const VectorXf& xi) const = 0;

  // Gradient and hessian of map from xi to x.
  virtual void GradientAndHessianXi(const VectorXf& xi, Eigen::Ref<VectorXf> grad,
                                    Eigen::Ref<MatrixXf> hess) const = 0;

  // Integrate these dynamics forward in time.
  // Options include integration for a single timestep, between arbitrary times,
  // and within a single timestep.
  VectorXf Integrate(Time time_interval, const VectorXf& xi0,
                     const std::vector<VectorXf>& vs) const;
  VectorXf Integrate(Time t0, Time t, const VectorXf& xi0,
                     const OperatingPoint& operating_point,
                     const std::vector<Strategy>& strategies) const;
  VectorXf Integrate(size_t initial_timestep, size_t final_timestep,
                     const VectorXf& xi0, const OperatingPoint& operating_point,
                     const std::vector<Strategy>& strategies) const;
  VectorXf IntegrateToNextTimeStep(
      Time t0, const VectorXf& xi0,
      const OperatingPoint& operating_point,
      const std::vector<Strategy>& strategies) const;
  VectorXf IntegrateFromPriorTimeStep(
      Time t, const VectorXf& xi0,
      const OperatingPoint& operating_point,
      const std::vector<Strategy>& strategies) const;

  // Getters.
  const LinearDynamicsApproximation& LinearizedSystem() const {
    if (!discrete_linear_system_) ComputeLinearizedSystem();
    return *discrete_linear_system_;
  }
  Dimension XDim() const { return xdim_; }
  Dimension TotalUDim() const {
    Dimension total = 0;
    for (PlayerIndex ii = 0; ii < NumPlayers(); ii++) total += UDim(ii);
    return total;
  }

  virtual Dimension UDim(PlayerIndex player_idx) const = 0;
  virtual PlayerIndex NumPlayers() const = 0;

 protected:
  MultiPlayerFlatSystem(Dimension xdim, Time time_step) 
        : xdim_(xdim), time_step_(time_step) {}

  // Discrete time approximation of the underlying linearized system.
  virtual void ComputeLinearizedSystem() const = 0;

  // State dimension.
  const Dimension xdim_;

  // Linearized system (discrete and continuous time).
  mutable std::unique_ptr<const LinearDynamicsApproximation> discrete_linear_system_;
  mutable std::unique_ptr<const LinearDynamicsApproximation> continuous_linear_system_;

  // Time step.
  const Time time_step_;
};  //\class MultiPlayerFlatSystem

}  // namespace ilqgames

#endif
