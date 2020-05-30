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

#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/solver/top_down_renderable_problem.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <imgui/imgui.h>
#include <vector>

namespace ilqgames {

class TopDownRenderer {
 public:
  ~TopDownRenderer() {}

  // Takes in a log and lists of x/y/heading indices in the state vector.
  TopDownRenderer(
      const std::shared_ptr<const ControlSliders>& sliders,
      const std::vector<std::shared_ptr<const TopDownRenderableProblem>>&
          problems)
      : sliders_(sliders),
        problems_(problems),
        center_delta_(0.0, 0.0),
        last_mouse_position_(0.0, 0.0),
        pixel_to_meter_ratio_(5.0) {
    CHECK_NOTNULL(sliders_.get());
    CHECK_EQ(problems_.size(), sliders_->NumProblems());
    for (const auto& problem : problems_) CHECK_NOTNULL(problem.get());
  }

  // Render the log in a top-down view.
  void Render();

 private:
  // Convert between positions/headings in Cartesian coordinates and window
  // coordinates.
  float CurrentZoomLevel() const;
  float LengthToPixels(float l) const { return l * CurrentZoomLevel(); }
  float PixelsToLength(float p) const { return p / CurrentZoomLevel(); }
  float HeadingToWindowCoordinates(float heading) const { return -heading; }
  ImVec2 PositionToWindowCoordinates(float x, float y) const;
  Point2 WindowCoordinatesToPosition(const ImVec2& coords) const;
  ImVec2 WindowCenter() const;

  // Control sliders.
  const std::shared_ptr<const ControlSliders> sliders_;

  // Renderable problem. Used to map from states along an operating point to
  // position and heading.
  const std::vector<std::shared_ptr<const TopDownRenderableProblem>> problems_;

  // Difference from center of the window in world coordinates.
  ImVec2 center_delta_;

  // Mouse position at most recent key press, in window coordinates.
  ImVec2 last_mouse_position_;

  // Zoom level. This will be the pixel-to-meter ratio.
  float pixel_to_meter_ratio_;
};  // class TopDownRenderer

}  // namespace ilqgames

#endif
