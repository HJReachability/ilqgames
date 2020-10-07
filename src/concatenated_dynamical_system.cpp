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
// Multi-player dynamical system comprised of several single player subsystems.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/utils/linear_dynamics_approximation.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>

namespace ilqgames {

ConcatenatedDynamicalSystem::ConcatenatedDynamicalSystem(
    const SubsystemList& subsystems)
    : MultiPlayerDynamicalSystem(std::accumulate(
          subsystems.begin(), subsystems.end(), 0,
          [](Dimension total,
             const std::shared_ptr<SinglePlayerDynamicalSystem>& subsystem) {
            CHECK_NOTNULL(subsystem.get());
            return total + subsystem->XDim();
          })),
      subsystems_(subsystems) {
  // Populate subsystem start dimensions.
  subsystem_start_dims_.push_back(0);
  for (const auto& subsystem : subsystems_) {
    subsystem_start_dims_.push_back(subsystem_start_dims_.back() +
                                    subsystem->XDim());
  }
}

VectorXf ConcatenatedDynamicalSystem::Evaluate(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate 'xdot' one subsystem at a time.
  VectorXf xdot(xdim_);
  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    xdot.segment(dims_so_far, subsystem->XDim()) = subsystem->Evaluate(
        t, x.segment(dims_so_far, subsystem->XDim()), us[ii]);
    dims_so_far += subsystem->XDim();
  }

  return xdot;
}

LinearDynamicsApproximation ConcatenatedDynamicalSystem::Linearize(
    Time t, const VectorXf& x, const std::vector<VectorXf>& us) const {
  CHECK_EQ(us.size(), NumPlayers());

  // Populate a block-diagonal A, as well as Bs.
  LinearDynamicsApproximation linearization(*this);

  Dimension dims_so_far = 0;
  for (size_t ii = 0; ii < NumPlayers(); ii++) {
    const auto& subsystem = subsystems_[ii];
    const Dimension xdim = subsystem->XDim();
    const Dimension udim = subsystem->UDim();
    subsystem->Linearize(
        t, x.segment(dims_so_far, xdim), us[ii],
        linearization.A.block(dims_so_far, dims_so_far, xdim, xdim),
        linearization.Bs[ii].block(dims_so_far, 0, xdim, udim));

    dims_so_far += subsystem->XDim();
  }

  return linearization;
}

float ConcatenatedDynamicalSystem::DistanceBetween(const VectorXf& x0,
                                                   const VectorXf& x1) const {
  // HACK: assumes only first subsystem matters.
  return subsystems_[0]->DistanceBetween(x0.head(subsystems_[0]->XDim()),
                                         x1.head(subsystems_[0]->XDim()));

  // Dimension dims_so_far = 0;
  // float total = 0.0;

  // // Accumulate total across all subsystems.
  // for (const auto& subsystem : subsystems_) {
  //   const Dimension xdim = subsystem->XDim();
  //   total += subsystem->DistanceBetween(x0.segment(dims_so_far, xdim),
  //                                       x1.segment(dims_so_far, xdim));

  //   dims_so_far += xdim;
  // }

  // return total;
}

std::vector<Dimension> ConcatenatedDynamicalSystem::PositionDimensions() const {
  std::vector<Dimension> dims;

  for (const auto& s : subsystems_) {
    const std::vector<Dimension> sub_dims = s->PositionDimensions();
    dims.insert(dims.end(), sub_dims.begin(), sub_dims.end());
  }

  return dims;
}

}  // namespace ilqgames
