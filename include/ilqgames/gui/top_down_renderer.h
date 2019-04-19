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
// Core renderer for 2D top-down trajectories. Integrates with DearImGui.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_GUI_TOP_DOWN_RENDERER_H
#define ILQGAMES_GUI_TOP_DOWN_RENDERER_H

#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <vector>

namespace ilqgames {

class TopDownRenderer {
 public:
  ~TopDownRenderer() {}
  TopDownRenderer(float meter_to_pixel_ratio, const std::shared_ptr<Log>& log)
      : meter_to_pixel_ratio_(meter_to_pixel_ratio), log_(log) {
    CHECK_NOTNULL(log_.get());
  }

  // Render the log in a top-down view.
  // Takes in lists of x/y/heading indices in the state vector.
  void Render(const std::vector<Dimension>& x_idxs,
              const std::vector<Dimension>& y_idxs,
              const std::vector<Dimension>& heading_idxs) const;

 private:
  // Convert between positions/headings in Cartesian coordinates and window
  // coordinates.
  ImVec2 PositionToWindowCoordinates(float x, float y) const;

  // Static variables for what time to show the state and which iterate to use.
  static float time_;
  static int iterate_;

  // Conversion between meters and pixels.
  const float meter_to_pixel_ratio;

  // Log to render.
  const std::shared_ptr<Log>& log_;
};  // class TopDownRenderer

}  // namespace ilqgames

#endif
