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
// Container to store an operating point, i.e. states and controls for each
// player.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_UTILS_OPERATING_POINT_H
#define ILQGAMES_UTILS_OPERATING_POINT_H

#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <vector>

namespace ilqgames {

struct OperatingPoint {
  // Time-indexed list of states.
  std::vector<VectorXf> xs;

  // Time-indexed list of controls for all players, i.e. us[kk] is the list of
  // controls for all players at time index kk.
  std::vector<std::vector<VectorXf>> us;

  // Initial time stamp.
  Time t0;

  // Construct with empty vectors of the right size, and optionally zero out if
  // dynamics is non-null.
  OperatingPoint(size_t num_time_steps, PlayerIndex num_players,
                 Time initial_time);

  template <typename MultiPlayerSystemType>
  OperatingPoint(size_t num_time_steps, Time initial_time,
                 const std::shared_ptr<const MultiPlayerSystemType>& dynamics)
      : OperatingPoint(num_time_steps, dynamics->NumPlayers(), initial_time) {
    CHECK_NOTNULL(dynamics.get());
    for (size_t kk = 0; kk < num_time_steps; kk++) {
      xs[kk] = VectorXf::Zero(dynamics->XDim());
      for (PlayerIndex ii = 0; ii < dynamics->NumPlayers(); ii++)
        us[kk][ii] = VectorXf::Zero(dynamics->UDim(ii));
    }
  }

  // Custom swap function.
  void swap(OperatingPoint& other);
};  // struct OperatingPoint

}  // namespace ilqgames

#endif
