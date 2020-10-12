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

#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/gui/top_down_renderer.h>
#include <ilqgames/utils/operating_point.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/types.h>

#include <imgui/imgui.h>
#include <math.h>
#include <algorithm>
#include <vector>

namespace ilqgames {

namespace {
// Zoom parameters.
static constexpr float kPixelsToZoomConversion = 1.0 / 20.0;
static constexpr float kMinZoom = 2.0;
}  // anonymous namespace

void TopDownRenderer::Render() {
  // Extract current log.
  const auto& logs = sliders_->LogForEachProblem();

  // Do nothing if no iterates yet.
  if (sliders_->MaxLogIndex() == 1) return;

  // Get the number of agents in each problem.
  std::vector<size_t> num_agents(problems_.size());
  std::transform(
      problems_.begin(), problems_.end(), num_agents.begin(),
      [](const std::shared_ptr<const TopDownRenderableProblem>& problem) {
        return problem->Dynamics()->NumPlayers();
      });

  // Set up main top-down viewer window.
  ImGui::Begin("Top-Down View");

  // Set up child window displaying key codes for navigation and zoom,
  // as well as mouse position.
  const ImVec2 mouse_position = ImGui::GetMousePos();
  if (ImGui::BeginChild("User Guide")) {
    ImGui::TextUnformatted("Press \"c\" key to enable navigation.");
    ImGui::TextUnformatted("Press \"z\" key to change zoom.");

    const Point2 mouse_point = WindowCoordinatesToPosition(mouse_position);
    ImGui::Text("Mouse is at: (%3.1f, %3.1f)", mouse_point.x(),
                mouse_point.y());
  }
  ImGui::EndChild();

  // Update last mouse position if "c" or "z" key was just pressed.
  constexpr bool kCatchRepeats = false;
  if (ImGui::IsKeyPressed(ImGui::GetIO().KeyMap[ImGuiKey_C], kCatchRepeats) ||
      ImGui::IsKeyPressed(ImGui::GetIO().KeyMap[ImGuiKey_Z], kCatchRepeats))
    last_mouse_position_ = mouse_position;
  else if (ImGui::IsKeyReleased(ImGui::GetIO().KeyMap[ImGuiKey_C])) {
    // When "c" is released, update center delta.
    center_delta_.x +=
        PixelsToLength(mouse_position.x - last_mouse_position_.x);
    center_delta_.y -=
        PixelsToLength(mouse_position.y - last_mouse_position_.y);
  } else if (ImGui::IsKeyReleased(ImGui::GetIO().KeyMap[ImGuiKey_Z])) {
    // When "z" is released, update pixel to meter ratio.
    const float mouse_delta_y = mouse_position.y - last_mouse_position_.y;
    pixel_to_meter_ratio_ =
        std::max(kMinZoom, pixel_to_meter_ratio_ -
                               kPixelsToZoomConversion * mouse_delta_y);
  }

  // Get the draw list for this window.
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  const ImU32 trajectory_color = ImColor(ImVec4(1.0, 1.0, 1.0, 0.5));
  const float trajectory_thickness = std::min(1.0f, LengthToPixels(0.5));

  // Loop over all problems and render one at a time.
  for (size_t problem_idx = 0; problem_idx < problems_.size(); problem_idx++) {
    const auto& problem = problems_[problem_idx];
    const auto& log = logs[problem_idx];

    // (1) Draw this trajectory iterate.
    std::vector<ImVec2> points(time::kNumTimeSteps);
    for (size_t ii = 0; ii < num_agents[problem_idx]; ii++) {
      for (size_t kk = 0; kk < time::kNumTimeSteps; kk++) {
        const VectorXf x = log->State(sliders_->SolverIterate(problem_idx), kk);
        points[kk] =
            PositionToWindowCoordinates(problem->Xs(x)[ii], problem->Ys(x)[ii]);
      }

      constexpr bool kPolylineIsClosed = false;
      draw_list->AddPolyline(points.data(), time::kNumTimeSteps,
                             trajectory_color, kPolylineIsClosed,
                             trajectory_thickness);
    }

    // Agent colors will all be greenish. Also specify circle radius and
    // triangle base and height (in pixels).
    constexpr float kMinGreen = 0.15;
    constexpr float kMaxGreen = 1.0 - kMinGreen;
    const float color_scaling = (1.0 - 2.0 * kMinGreen) *
                                static_cast<float>(problem_idx + 1) /
                                static_cast<float>(problems_.size());

    const ImU32 agent_color =
        ImColor(ImVec4(0.15, kMinGreen + color_scaling,
                       kMinGreen + kMaxGreen - color_scaling, 1.0));
    const float agent_radius = std::max(5.0f, LengthToPixels(2.5));
    const float agent_base = std::max(6.0f, LengthToPixels(2.5));
    const float agent_height = std::max(10.0f, LengthToPixels(3.0));

    // Draw each position as either an isosceles triangle (if heading idx is
    // >= 0) or a circle (if heading idx < 0).
    const VectorXf current_x =
        log->InterpolateState(sliders_->SolverIterate(problem_idx),
                              sliders_->InterpolationTime(problem_idx));
    const std::vector<float> current_pxs = problem->Xs(current_x);
    const std::vector<float> current_pys = problem->Ys(current_x);
    const std::vector<float> current_thetas = problem->Thetas(current_x);
    for (size_t ii = 0; ii < num_agents[problem_idx]; ii++) {
      const ImVec2 p =
          PositionToWindowCoordinates(current_pxs[ii], current_pys[ii]);

      if (ii >= current_thetas.size()) {
        VLOG(2) << "More players than coordinates to visualize.";
        break;
      } else {
        const float heading = HeadingToWindowCoordinates(current_thetas[ii]);
        const float cheading = std::cos(heading);
        const float sheading = std::sin(heading);

        // Triangle vertices (top, bottom left, bottom right in Frenet frame).
        // NOTE: this may not be in CCW order. Not sure if that matters.
        const ImVec2 top(p.x + agent_height * cheading,
                         p.y + agent_height * sheading);
        const ImVec2 bl(p.x - 0.5 * agent_base * sheading,
                        p.y + 0.5 * agent_base * cheading);
        const ImVec2 br(p.x + 0.5 * agent_base * sheading,
                        p.y - 0.5 * agent_base * cheading);

        draw_list->AddTriangleFilled(bl, br, top, agent_color);
        draw_list->AddCircle(p, agent_radius, agent_color);
      }
    }
  }

  ImGui::End();
}

inline float TopDownRenderer::CurrentZoomLevel() const {
  float conversion = pixel_to_meter_ratio_;

  // Handle "z" down case.
  if (ImGui::IsKeyDown(ImGui::GetIO().KeyMap[ImGuiKey_Z])) {
    const float mouse_delta_y = ImGui::GetMousePos().y - last_mouse_position_.y;
    conversion = std::max(kMinZoom,
                          conversion - kPixelsToZoomConversion * mouse_delta_y);
  }

  return conversion;
}

inline ImVec2 TopDownRenderer::PositionToWindowCoordinates(float x,
                                                           float y) const {
  ImVec2 coords = WindowCenter();

  // Offsets if "c" key is currently down.
  if (ImGui::IsKeyDown(ImGui::GetIO().KeyMap[ImGuiKey_C])) {
    const ImVec2 mouse_position = ImGui::GetMousePos();
    x += PixelsToLength(mouse_position.x - last_mouse_position_.x);
    y -= PixelsToLength(mouse_position.y - last_mouse_position_.y);
  }

  coords.x += LengthToPixels(x + center_delta_.x);
  coords.y -= LengthToPixels(y + center_delta_.y);
  return coords;
}

inline Point2 TopDownRenderer::WindowCoordinatesToPosition(
    const ImVec2& coords) const {
  const ImVec2 center = WindowCenter();

  // NOTE: only correct when "c" key is not down.
  const float x = PixelsToLength(coords.x - center.x) - center_delta_.x;
  const float y = PixelsToLength(center.y - coords.y) - center_delta_.y;
  return Point2(x, y);
}

inline ImVec2 TopDownRenderer::WindowCenter() const {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const float window_width = ImGui::GetWindowWidth();
  const float window_height = ImGui::GetWindowHeight();

  const float center_x = window_pos.x + 0.5 * window_width;
  const float center_y = window_pos.y + 0.5 * window_height;
  return ImVec2(center_x, center_y);
}

}  // namespace ilqgames
