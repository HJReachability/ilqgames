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
// Set the position dimensions of an operating point to follow a given route
// polyline.
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/geometry/polyline2.h>
#include <ilqgames/utils/initialize_along_route.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <tuple>
#include <vector>

namespace ilqgames {

void InitializeAlongRoute(const Polyline2& route, float initial_route_pos,
                          float nominal_speed,
                          const std::pair<Dimension, Dimension>& position_dims,
                          OperatingPoint* operating_point) {
  CHECK_NOTNULL(operating_point);
  CHECK(!operating_point->xs.empty());
  CHECK_GT(operating_point->xs[0].size(), 0);

  // Loop through each time step and determine where we should be.
  for (size_t kk = 0; kk < operating_point->xs.size(); kk++) {
    const float route_pos = initial_route_pos + nominal_speed *
                                                    static_cast<Time>(kk) *
                                                    time::kTimeStep;

    const Point2 route_pt = route.PointAt(route_pos);
    operating_point->xs[kk](position_dims.first) = route_pt.x();
    operating_point->xs[kk](position_dims.second) = route_pt.y();
  }
}

}  // namespace ilqgames
