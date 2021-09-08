%% collision_avoidance_example();
%% receding_horizon_example();
%% air_3d_example(true);
minimally_invasive_example();


function air_3d_example(baseline)
%% Grid
grid_min = [-10; -10; -pi]; % Lower corner of computation domain
grid_max = [10; 10; pi];    % Upper corner of computation domain
N = [41; 41; 21];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);

%% Compute optimal trajectory from some initial state
xinit = [3, 4, -2*pi/3];
%xinit = [1, 0.5, -2*pi/3];

%% target set
R = 1.0;
data0 = -shapeCylinder(g, 3, [0; 0; 0], R);

%% time vector
t0 = 0;
tMax = 2;
dt = 0.1;
tau = t0:dt:tMax;

if (baseline)
  %% problem parameters
  %% input bounds
  uMax = 1;

  %% Speeds.
  eSpeed = 1.0;
  pSpeed = 0.1;

  %% control trying to min or max value function?
  eMode = 'min';
  pMode = 'max';

  %% Pack problem parameters
  %% Define dynamic system
  dAir = Air3D([0; 0; 0], uMax, uMax, eSpeed, pSpeed);

  %% Put grid and dynamic systems into schemeData
  schemeData.grid = g;
  schemeData.dynSys = dAir;
  schemeData.accuracy = 'high'; %set accuracy
  schemeData.uMode = eMode;
  schemeData.dMode = pMode;

  %% Compute value function
  HJIextraArgs.visualize = true; %show plot
  HJIextraArgs.fig_num = 1; %set figure number
  HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update

  [data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, 'maxVOverTime', HJIextraArgs);

  dAir.x = xinit;
  TrajextraArgs.uMode = eMode; %set if control wants to min or max
  TrajextraArgs.dMode = pMode; %set if disturbance wants to min or max
  TrajextraArgs.visualize = true; %show plot
  TrajextraArgs.fig_num = 2; %figure number

  %%we want to see the first two dimensions (x and y)
  TrajextraArgs.projDim = [1 1 0];

  %%flip data time points so we start from the beginning of time
  dataTraj = flip(data, 4);

  [traj, traj_tau] = computeOptTraj(g, dataTraj, tau2, dAir, TrajextraArgs);
  traj = traj'; % Transpose traj to have colums be different timesteps
  value = eval_u(g, data(:,:,:,end), xinit)
  traj
end

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
figure;
title(sprintf('Sensitivity to Scale ($\\epsilon = %1.2f$)', 1.0), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
if baseline
  plot(traj(:, 1), traj(:, 2), 'g-o', 'DisplayName', 'Best-effort solution');
  value_format_string = '\\tilde V(x_1) - V(x_1) = %1.2f$';
end

if ~baseline
  value = 0.0;
  value_format_string = "\\tilde V(x_1) = %1.2f$";
end

if false
x0_flag = "--px0=" + xinit(1) + " --py0=" + xinit(2) + " --theta0=" + xinit(3) + " --v0=" + xinit(4);
%%distance_traveled = xinit(4) * tMax * 0.5;
%%value_to_add = R - sqrt((xinit(1) + distance_traveled * cos(xinit(3))).^2 + ...
%%                        (xinit(2) + distance_traveled * sin(xinit(3))).^2);
value_to_add = R + xinit(2);

for a = scale_vals
  [ilq_traj, values] = run_ilqgames("two_player_reachability_example", "", ...
                                    a, nominal_control_penalty, x0_flag);
  plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', 'color', colormap(a, scale_vals, true), ...
       'DisplayName', sprintf(char("$a = %1.2f, " + value_format_string), a, ...
                              values(1) + value + value_to_add));
end

hold off;
l1 = legend('Location', 'SouthWest');

figure;
title(sprintf('Sensitivity to Control Penalty ($a = %1.2f$)', nominal_scale), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
if baseline
  plot(traj(:, 1), traj(:, 2), 'g-o', 'DisplayName', 'Best-effort solution');
end

for epsilon = control_penalty_vals
  [ilq_traj, values] = run_ilqgames("two_player_reachability_example", "", ...
                                    nominal_scale, epsilon, x0_flag);
  plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', 'color', ...
       colormap(epsilon, control_penalty_vals, false), 'DisplayName', ...
       sprintf(char("$\\epsilon = %1.2f, " + value_format_string), epsilon, ...
               values(1) + value + value_to_add));
end

hold off;
l2 = legend('Location', 'SouthWest');

set(l1, 'Interpreter', 'latex');
set(l2, 'Interpreter', 'latex');

%% Reachable set plot.
make_surf_plot = false;
if make_surf_plot
  tilde_V = zeros(N([1, 2])');
  nominal_theta = pi / 2;
  nominal_v = 1.0;
  theta_idx = int64((nominal_theta - grid_min(3)) / g.dx(3));
  v_idx = int64((nominal_v - grid_min(4)) / g.dx(4));

  for x_idx = 1:N(1)
    for y_idx = 1:N(2)
      %% Unpack state.
      x0 = [g.xs{1}(x_idx, y_idx, theta_idx, v_idx), g.xs{2}(x_idx, y_idx, theta_idx, v_idx), ...
            nominal_theta, nominal_v];
      x0_flag = "--px0=" + x0(1) + " --py0=" + x0(2) + " --theta0=" + x0(3) + " --v0=" + x0(4);
      distance_traveled = x0(4) * tMax * 0.5;
      value_to_add = R - sqrt((x0(1) + distance_traveled * cos(x0(3))).^2 + ...
                              (x0(2) + distance_traveled * sin(x0(3))).^2);

      [ilq_traj, values] = run_ilqgames("two_player_reachability_example", "", ...
                                        nominal_scale, nominal_control_penalty, x0_flag);
      tilde_V(x_idx, y_idx) = values(1) + value_to_add;
    end
  end

  figure;
  title(sprintf('Comparison of Value Functions $(a = %1.2f, \\epsilon = %1.2f)$', ...
                nominal_scale, nominal_control_penalty), 'Interpreter', 'latex');
  xlabel('$p_x$ (m)', 'Interpreter', 'latex');
  ylabel('$p_y$ (m)', 'Interpreter', 'latex');
  zlabel('Value');
  hold on;
  if (baseline)
    s1 = surf(g.xs{1}(:, :, theta_idx, v_idx), g.xs{2}(:, :, theta_idx, v_idx), ...
              -data0(:,:,theta_idx,v_idx,end), 'FaceColor', 'green', 'FaceAlpha', ...
              0.5, 'EdgeColor', 'none');
  end

  s2 = surf(g.xs{1}(:, :, theta_idx, v_idx), g.xs{2}(:, :, theta_idx, v_idx), tilde_V);
  hold off;

  if baseline
    l3 = legend([s1, s2], {'$V(x_1)$', '$\tilde V(x_1)$'});
  else
    l3 = legend([s2], {'$\tilde V(x_1)$'});
  end

  set(l3, 'Interpreter', 'latex');
end
end
end

function collision_avoidance_example()
close all;

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
regularization_vals = linspace(0.01, 1, 5);

x0_flag = " --d0=5 --v0=5";

figure;
set(gca, 'FontSize', 24');
title('Sensitivity to Regularization', 'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');
xlim([-5.5, 5.5]);
ylim([-5.5, 5.5]);

hold on;
ii = 1;
for regularization = regularization_vals
  [ilq_traj, values] = run_ilqgames("three_player_collision_avoidance_reachability_example", "", ...
                                    regularization, x0_flag);
  if max(abs(values(1) - values(2)), abs(values(1) - values(3))) > 0.01
    disp(char("Error! Values do not match: " + values(1) + ", " + values(2) + ", " + values(3)));
  end

  pe(ii) = plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', ...
                'Color', colormap(regularization, regularization_vals, false), ...
                'DisplayName', sprintf('$\\epsilon = %1.2f$', regularization));
  plot(ilq_traj(:, 6), ilq_traj(:, 7), 'x:', 'color', ...
       colormap(regularization, regularization_vals, false));
  plot(ilq_traj(:, 11), ilq_traj(:, 12), 'x--', 'color', ...
       colormap(regularization, regularization_vals, false));

  ii = ii + 1;
end

hold off;
l2 = legend(pe, 'Location', 'NorthEast');

set(l2, 'Interpreter', 'latex');
end

%% Simple red-blue colormap.
function color = colormap(val, opts, reverse)
  r = (val - opts(1)) / (opts(end) - opts(1));
  color = [r, 0.25, 1.0 - r];

  if reverse
    color = [1.0 - r, 0.25, r];
  end
end

%% Compute ILQ trajectory for given example.
function [traj, values] = run_ilqgames(exec, extra_suffix, regularization, x0_flag)
  experiment_name = exec + "_" + regularization + x0_flag;
  experiment_arg = " --experiment_name='" + experiment_name + extra_suffix + "'";
  save_flag = "--save" + extra_suffix;

  if ~experiment_already_run(char(experiment_name + extra_suffix))
    %% Stitch together the command for the executable.
    instruction = "../bin/" + exec + " --noviz " + save_flag + ...
                  " --last_traj" + experiment_arg + ...
                  " --convergence_tolerance=0.02 --initial_alpha_scaling=0.9" + ...
                  " --regularization=" + regularization + ...
                  x0_flag;
    system(char(instruction));
  end

  log_folder = "../logs/";
  cd(char(log_folder + experiment_name + extra_suffix));
  dirs = dir;
  last_iterate = "blah";
  for ii = 1:size(dirs)
      if dirs(ii).name(1) ~= '.'
          last_iterate = dirs(ii).name;
          break;
      end
  end
  cd('../../matlab');

  traj = load(log_folder + experiment_name + extra_suffix + "/" + ...
              last_iterate + "/xs.txt");
  values = load(log_folder + experiment_name + extra_suffix + "/" + ...
                last_iterate + "/costs.txt");
%%  for ii = 1:length(values)
%%    values(ii) = log(values(ii)) / scale;
%%  end
end

function minimally_invasive_example()
  close all;

  exec = "minimally_invasive_intersection_example";
  experiment_name = "minimally_invasive_example";
  regularization = 0.01;

  if ~experiment_already_run(char(experiment_name + "_safety"))
    %% Stitch together the command for the executable.
    instruction = "../bin/" + exec + " --noviz " + "--save" + ...
                  " --last_traj --experiment_name=" + experiment_name + ...
                  " --regularization=" + regularization;
    system(char(instruction));
  end

  figure;
  set(gca, 'FontSize', 16');

  dt = 0.1;
  iters = 10:2:16;
  for ii = 1:numel(iters)
    iter = iters(ii);
    subplot(1, numel(iters), ii);

    [t0, original_xs, ~] = unpack_log(experiment_name + "_original", iter);
    [~, safety_xs, safety_V1] = unpack_log(experiment_name + "_safety", iter);

    %% HACK: safety controller is active if V1 is above threshold.
    safety_threshold = -1;
    if (safety_V1 > safety_threshold)
      safety_active = "Safety";
    else
      safety_active = "Nominal";
    end

    title(sprintf('$t_0 = %4.2f$ (s), %s', t0, safety_active), ...
          'Interpreter', 'latex');
    xlabel('$p_x$ (m)', 'Interpreter', 'latex');
    ylabel('$p_y$ (m)', 'Interpreter', 'latex');
    xlim([-12, 10]);
    ylim([-10, 40]);

    hold on;
    plot(original_xs(:, 1), original_xs(:, 2), 'r');
    plot(original_xs(:, 6), original_xs(:, 7), 'g');
    plot(original_xs(:, 11), original_xs(:, 12), 'b');
    plot(safety_xs(:, 1), safety_xs(:, 2), 'r--');
    plot(safety_xs(:, 6), safety_xs(:, 7), 'g--');
    plot(safety_xs(:, 11), safety_xs(:, 12), 'b--');
    hold off;
  end

  end

function [t0, xs, V1] = unpack_log(experiment_name, iter)
  assert(experiment_already_run(experiment_name));

  log_folder = "../logs/";
  cd(char(log_folder + experiment_name + "/" + iter));
  dirs = dir;
  last_iterate = dirs(end).name;
  cd('../../../matlab');

  t0 = load(log_folder + experiment_name + "/" + iter + "/" + ...
            last_iterate + "/t0.txt");
  xs = load(log_folder + experiment_name + "/" + iter + "/" + ...
            last_iterate + "/xs.txt");

  %% Get value for player 1.
  Vs = load(log_folder + experiment_name + "/" + iter + "/" + ...
            last_iterate + "/costs.txt");
  V1 = Vs(1);
end

function exists = experiment_already_run(folder_name)
  %% Get experiment folder names
  cd('../logs');
  dirs = dir;
  exists = false;
  for ii = 1:length(dirs)
    if strcmp(folder_name, dirs(ii).name)
      exists = true;
      break;
    end
  end
  cd('../matlab');
end
