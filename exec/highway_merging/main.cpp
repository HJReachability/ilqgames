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
// Main GUI for highway merging example.
// Formerly: Main GUI for three player intersection example.
//
///////////////////////////////////////////////////////////////////////////////

// #include <ilqgames/examples/three_player_overtaking_example.h>
#include <ilqgames/examples/highway_merging_example.h>
#include <ilqgames/gui/control_sliders.h>
#include <ilqgames/gui/cost_inspector.h>
#include <ilqgames/gui/top_down_renderer.h>
#include <ilqgames/solver/augmented_lagrangian_solver.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/check_local_nash_equilibrium.h>
#include <ilqgames/utils/solver_log.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>
#include <iostream>
#include <memory>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// Optional log saving and visualization.
DEFINE_bool(open_loop, false, "Use open loop (vs. feedback) solver.");
DEFINE_bool(save, false, "Optionally save solver logs to disk.");
DEFINE_bool(viz, true, "Visualize results in a GUI.");
DEFINE_bool(last_traj, false,
            "Should the solver only dump the last trajectory?");
DEFINE_string(experiment_name, "", "Name for the experiment.");

// Linesearch parameters.
DEFINE_bool(linesearch, true, "Should the solver linesearch?");
DEFINE_double(initial_alpha_scaling, 0.75, "Initial step size in linesearch.");
DEFINE_double(convergence_tolerance, 0.25, "L_inf tolerance for convergence.");
DEFINE_double(expected_decrease, 0.1, "KKT sq err expected decrease per iter.");

// Adversarial Time(s).
DEFINE_double(adversarial_time, 0.0,
              "Amount of time other agents are assumed to be adversarial");
DEFINE_double(
    Tadv_1, 0.0,
    "Amount of time other agents are assumed to be adversarial, Trajectory 1");
DEFINE_double(
    Tadv_2, 1.0,
    "Amount of time other agents are assumed to be adversarial, Trajectory 2");
DEFINE_double(
    Tadv_3, 2.0,
    "Amount of time other agents are assumed to be adversarial, Trajectory 3");

DEFINE_int32(unconstrained_solver_max_iters, 10,
             "Maximum iterations run by unconstrained solver");
DEFINE_double(geometric_mu_scaling, 1.1, "geometric mu scaling");
DEFINE_double(geometric_mu_downscaling, 0.5, "geometric mu downscaling");
DEFINE_double(geometric_lambda_downscaling, 0.5,
              "geometric lambda downscaling");
DEFINE_double(constraint_error_tolerance, 0.1, "constraint error tolerance");

DEFINE_bool(multi_traj, false,
            "Should the GUI print out trajectories corresponding to multiple "
            "adversarial times, at once?");

// About OpenGL function loaders: modern OpenGL doesn't have a standard header
// file and requires individual function pointers to be loaded manually. Helper
// libraries are often used for this purpose! Here we are supporting a few
// common ones: gl3w, glew, glad. You may use another loader/header of your
// choice (glext, glLoadGen, etc.), or chose to manually implement your own.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>  // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>  // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions.
#include <GLFW/glfw3.h>

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv) {
  const std::string log_file =
      ILQGAMES_LOG_DIR + std::string("/highway_merging.log");
  google::SetLogDestination(0, log_file.c_str());
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = true;

  // Set up the game.
  ilqgames::SolverParams params;
  params.open_loop = FLAGS_open_loop;
  params.max_backtracking_steps = 100;
  params.max_solver_iters = 100;
  params.unconstrained_solver_max_iters = 10;
  params.linesearch = FLAGS_linesearch;
  params.expected_decrease_fraction = FLAGS_expected_decrease;
  params.initial_alpha_scaling = FLAGS_initial_alpha_scaling;
  params.convergence_tolerance = FLAGS_convergence_tolerance;

  params.unconstrained_solver_max_iters = FLAGS_unconstrained_solver_max_iters;
  params.geometric_mu_scaling = FLAGS_geometric_mu_scaling;
  params.geometric_mu_downscaling = FLAGS_geometric_mu_downscaling;
  params.geometric_lambda_downscaling = FLAGS_geometric_lambda_downscaling;
  params.constraint_error_tolerance = FLAGS_constraint_error_tolerance;

  if (FLAGS_multi_traj) {
    // Create problem_00, problem_10, problem_20, with adversarial_time = 0.0,
    // 1.0, 2.0, respectively.
    auto problem_00 =
        std::make_shared<ilqgames::HighwayMergingExample>(FLAGS_Tadv_1);
    problem_00->Initialize();
    ilqgames::ILQSolver solver_00(problem_00, params);

    auto problem_10 =
        std::make_shared<ilqgames::HighwayMergingExample>(FLAGS_Tadv_2);
    problem_10->Initialize();
    ilqgames::ILQSolver solver_10(problem_10, params);

    auto problem_20 =
        std::make_shared<ilqgames::HighwayMergingExample>(FLAGS_Tadv_3);
    problem_20->Initialize();
    ilqgames::ILQSolver solver_20(problem_20, params);

    // Solve the game.
    const auto start_00 = std::chrono::system_clock::now();
    std::shared_ptr<const ilqgames::SolverLog> log_00 = solver_00.Solve();
    const std::vector<std::shared_ptr<const ilqgames::SolverLog>> logs_00 = {
        log_00};
    LOG(INFO) << "Solver completed in "
              << std::chrono::duration<ilqgames::Time>(
                     std::chrono::system_clock::now() - start_00)
                     .count()
              << " seconds.";

    const auto start_10 = std::chrono::system_clock::now();
    std::shared_ptr<const ilqgames::SolverLog> log_10 = solver_10.Solve();
    const std::vector<std::shared_ptr<const ilqgames::SolverLog>> logs_10 = {
        log_10};
    LOG(INFO) << "Solver completed in "
              << std::chrono::duration<ilqgames::Time>(
                     std::chrono::system_clock::now() - start_10)
                     .count()
              << " seconds.";

    const auto start_20 = std::chrono::system_clock::now();
    std::shared_ptr<const ilqgames::SolverLog> log_20 = solver_20.Solve();
    const std::vector<std::shared_ptr<const ilqgames::SolverLog>> logs_20 = {
        log_20};
    LOG(INFO) << "Solver completed in "
              << std::chrono::duration<ilqgames::Time>(
                     std::chrono::system_clock::now() - start_20)
                     .count()
              << " seconds.";

    // // Check if solution satisfies sufficient conditions for being a local
    // Nash. problem_00->OverwriteSolution(log_00->FinalOperatingPoint(),
    //                               log_00->FinalStrategies());
    // const bool is_local_nash_00 =
    //     CheckSufficientLocalNashEquilibrium(*problem_00);
    // if (is_local_nash_00)
    //   LOG(INFO) << "Solution is a local Nash.";
    // else
    //   LOG(INFO) << "Solution may not be a local Nash.";

    // problem_10->OverwriteSolution(log_10->FinalOperatingPoint(),
    //                               log_10->FinalStrategies());
    // const bool is_local_nash_10 =
    //     CheckSufficientLocalNashEquilibrium(*problem_10);
    // if (is_local_nash_10)
    //   LOG(INFO) << "Solution is a local Nash.";
    // else
    //   LOG(INFO) << "Solution may not be a local Nash.";

    // problem_20->OverwriteSolution(log_20->FinalOperatingPoint(),
    //                               log_20->FinalStrategies());
    // const bool is_local_nash_20 =
    //     CheckSufficientLocalNashEquilibrium(*problem_20);
    // if (is_local_nash_20)
    //   LOG(INFO) << "Solution is a local Nash.";
    // else
    //   LOG(INFO) << "Solution may not be a local Nash.";

    // // Confirm with numerical check.
    // constexpr float kMaxPerturbation = 0.1;
    // constexpr bool kOpenLoop = false;
    // problem_00->OverwriteSolution(log_00->FinalOperatingPoint(),
    //                               log_00->FinalStrategies());
    // const bool is_numerical_nash_00 = NumericalCheckLocalNashEquilibrium(
    //     *problem_00, kMaxPerturbation, kOpenLoop);
    // if (is_numerical_nash_00)
    //   LOG(INFO) << "Solution is a numerical Nash.";
    // else
    //   LOG(INFO) << "Solution is not a numerical Nash.";

    // problem_10->OverwriteSolution(log_10->FinalOperatingPoint(),
    //                               log_10->FinalStrategies());
    // const bool is_numerical_nash_10 = NumericalCheckLocalNashEquilibrium(
    //     *problem_10, kMaxPerturbation, kOpenLoop);
    // if (is_numerical_nash_10)
    //   LOG(INFO) << "Solution is a numerical Nash.";
    // else
    //   LOG(INFO) << "Solution is not a numerical Nash.";

    // problem_20->OverwriteSolution(log_20->FinalOperatingPoint(),
    //                               log_20->FinalStrategies());
    // const bool is_numerical_nash_20 = NumericalCheckLocalNashEquilibrium(
    //     *problem_20, kMaxPerturbation, kOpenLoop);
    // if (is_numerical_nash_20)
    //   LOG(INFO) << "Solution is a numerical Nash.";
    // else
    //   LOG(INFO) << "Solution is not a numerical Nash.";

    // // Dump the logs and/or exit.
    // if (FLAGS_save) {
    //   if (FLAGS_experiment_name == "") {
    //     CHECK(log_00->Save(FLAGS_last_traj));
    //   } else {
    //     CHECK(log_00->Save(FLAGS_last_traj, FLAGS_experiment_name));
    //   }
    // }
    // if (!FLAGS_viz) return 0;

    // if (FLAGS_save) {
    //   if (FLAGS_experiment_name == "") {
    //     CHECK(log_10->Save(FLAGS_last_traj));
    //   } else {
    //     CHECK(log_10->Save(FLAGS_last_traj, FLAGS_experiment_name));
    //   }
    // }
    // if (!FLAGS_viz) return 0;

    // if (FLAGS_save) {
    //   if (FLAGS_experiment_name == "") {
    //     CHECK(log_20->Save(FLAGS_last_traj));
    //   } else {
    //     CHECK(log_20->Save(FLAGS_last_traj, FLAGS_experiment_name));
    //   }
    // }
    // if (!FLAGS_viz) return 0;

    // Create a top-down renderer, control sliders, and cost inspector.
    std::shared_ptr<ilqgames::ControlSliders> sliders(
        new ilqgames::ControlSliders({logs_00, logs_10, logs_20}));
    ilqgames::TopDownRenderer top_down_renderer(
        sliders, {problem_00, problem_10, problem_20});
    ilqgames::CostInspector cost_inspector(
        sliders, {problem_00->PlayerCosts(), problem_10->PlayerCosts(),
                  problem_20->PlayerCosts()});

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

      // Decide GL+GLSL versions.
#if __APPLE__
    // GL 3.2 + GLSL 150.
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Required on Mac
#else
    // GL 3.0 + GLSL 130.
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
    // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(
        1280, 720, "ILQGames: Highway Merging Example", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#else
    bool err =
        false;  // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader
                // is likely to requires some form of initialization.
#endif
    if (err) {
      fprintf(stderr, "Failed to initialize OpenGL loader!\n");
      return 1;
    }

    // Setup Dear ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup Dear ImGui style.
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();

    // Background color.
    const ImVec4 clear_color =
        ImVec4(213.0 / 255.0, 216.0 / 255.0, 226.0 / 255.0, 1.0f);

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
      // Poll and handle events (inputs, window resize, etc.).
      glfwPollEvents();

      // Start the Dear ImGui frame.
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Control sliders.
      sliders->Render();

      // Top down view.
      top_down_renderer.Render();

      // Cost inspector.
      cost_inspector.Render();

      // Rendering
      ImGui::Render();
      int display_w, display_h;
      glfwMakeContextCurrent(window);
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwMakeContextCurrent(window);
      glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

  } else {
    auto problem = std::make_shared<ilqgames::HighwayMergingExample>(
        FLAGS_adversarial_time);
    problem->Initialize();
    ilqgames::AugmentedLagrangianSolver solver(problem, params);

    // Solve the game.
    const auto start = std::chrono::system_clock::now();
    std::shared_ptr<const ilqgames::SolverLog> log = solver.Solve();
    const std::vector<std::shared_ptr<const ilqgames::SolverLog>> logs = {log};
    LOG(INFO) << "Solver completed in "
              << std::chrono::duration<ilqgames::Time>(
                     std::chrono::system_clock::now() - start)
                     .count()
              << " seconds.";

    // Check if solution satisfies sufficient conditions for being a local Nash.
    problem->OverwriteSolution(log->FinalOperatingPoint(),
                               log->FinalStrategies());
    const bool is_local_nash = CheckSufficientLocalNashEquilibrium(*problem);
    if (is_local_nash)
      LOG(INFO) << "Solution is a local Nash.";
    else
      LOG(INFO) << "Solution may not be a local Nash.";

    // Confirm with numerical check.
    constexpr float kMaxPerturbation = 0.1;
    constexpr bool kOpenLoop = false;
    problem->OverwriteSolution(log->FinalOperatingPoint(),
                               log->FinalStrategies());
    const bool is_numerical_nash = NumericalCheckLocalNashEquilibrium(
        *problem, kMaxPerturbation, kOpenLoop);
    if (is_numerical_nash)
      LOG(INFO) << "Solution is a numerical Nash.";
    else
      LOG(INFO) << "Solution is not a numerical Nash.";

    // Dump the logs and/or exit.
    if (FLAGS_save) {
      if (FLAGS_experiment_name == "") {
        CHECK(log->Save(FLAGS_last_traj));
      } else {
        CHECK(log->Save(FLAGS_last_traj, FLAGS_experiment_name));
      }
    }
    if (!FLAGS_viz) return 0;

    // Create a top-down renderer, control sliders, and cost inspector.
    std::shared_ptr<ilqgames::ControlSliders> sliders(
        new ilqgames::ControlSliders({logs}));
    ilqgames::TopDownRenderer top_down_renderer(sliders, {problem});
    ilqgames::CostInspector cost_inspector(sliders, {problem->PlayerCosts()});

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

      // Decide GL+GLSL versions.
#if __APPLE__
    // GL 3.2 + GLSL 150.
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Required on Mac
#else
    // GL 3.0 + GLSL 130.
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
    // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(
        1280, 720, "ILQGames: Highway Merging Example", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#else
    bool err =
        false;  // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader
                // is likely to requires some form of initialization.
#endif
    if (err) {
      fprintf(stderr, "Failed to initialize OpenGL loader!\n");
      return 1;
    }

    // Setup Dear ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup Dear ImGui style.
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();

    // Background color.
    const ImVec4 clear_color =
        ImVec4(213.0 / 255.0, 216.0 / 255.0, 226.0 / 255.0, 1.0f);

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
      // Poll and handle events (inputs, window resize, etc.).
      glfwPollEvents();

      // Start the Dear ImGui frame.
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Control sliders.
      sliders->Render();

      // Top down view.
      top_down_renderer.Render();

      // Cost inspector.
      cost_inspector.Render();

      // Rendering
      ImGui::Render();
      int display_w, display_h;
      glfwMakeContextCurrent(window);
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwMakeContextCurrent(window);
      glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
  }
  return 0;
}
