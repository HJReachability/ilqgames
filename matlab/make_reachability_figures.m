%% Script to compare ILQ with grid-based HJ reachability computation, and in larger
%% example where comparison is not possible just approximate the reach set.

%one_player_comparison(false);
%two_player_comparison(false);
collision_avoidance_example();

function one_player_comparison(baseline)
% Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory


%% Compute optimal trajectory from some initial state
%set the initial state
xinit = [1.75, 1.75, 0];

if baseline
%% Grid
grid_min = [-4; -4; -pi]; % Lower corner of computation domain
grid_max = [4; 4; pi];    % Upper corner of computation domain
N = [41; 41; 41];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 3.0;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = -shapeCylinder(g, 3, [0; 0; 0], R);
% also try shapeRectangleByCorners, shapeSphere, etc.

%% time vector
t0 = 0;
tMax = 2;
dt = 0.1;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
speed = 1;
wMax = 1;
% do dStep1 here

% control trying to min or max value function?
uMode = 'max';
% do dStep2 here


%% Pack problem parameters

% Define dynamic system
% obj = DubinsCar(x, wMax, speed, dMax)
dCar = DubinsCar([0, 0, 0], wMax, speed); %do dStep3 here

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
%do dStep4 here

%% additive random noise
%do Step8 here
%HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
% Try other noise coefficients, like:
%    [0.2; 0; 0]; % Noise on X state
%    [0.2,0,0;0,0.2,0;0,0,0.5]; % Independent noise on all states
%    [0.2;0.2;0.5]; % Coupled noise on all states
%    {zeros(size(g.xs{1})); zeros(size(g.xs{1})); (g.xs{1}+g.xs{2})/20}; % State-dependent noise

%% If you have obstacles, compute them here

%% Compute value function

HJIextraArgs.visualize = false; %show plot
HJIextraArgs.fig_num = 1; %set figure number
HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);

%%check if this initial state is in the BRS/BRT
%value = eval_u(g, data, x)
value = eval_u(g,data(:,:,:,end),xinit);

dCar.x = xinit; %set initial state of the dubins car

TrajextraArgs.uMode = uMode; %set if control wants to min or max
TrajextraArgs.visualize = false; %show plot
TrajextraArgs.fig_num = 2; %figure number

%%we want to see the first two dimensions (x and y)
TrajextraArgs.projDim = [1 1 0];

%%flip data time points so we start from the beginning of time
dataTraj = flip(data,4);

[traj, traj_tau] = computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);
traj = traj'; % Transpose traj to have colums be different timesteps
value = eval_u(g, data(:,:,:,end), xinit);
end

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
scale_vals = linspace(0.1, 1.0, 5);
control_penalty_vals = linspace(1.0, 2.0, 5);

nominal_scale = 0.5;
nominal_control_penalty = 1.0;

figure;
title(sprintf('Sensitivity to Scale ($\\epsilon = %1.2f$)', nominal_control_penalty), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
if baseline
  plot(traj(:, 1), traj(:, 2), 'g-o', 'DisplayName', 'Best-effort solution');
  value_format_string = '\\tilde V(x_1) - V(x_1) = %1.2f';
end

if ~baseline
  value = 0.0;
  value_format_string = "\\tilde V(x_1) = %1.2f$";
end

x0_flag = "--px0=" + xinit(1) + " --py0=" + xinit(2) + " --theta0=" + xinit(3);

for a = scale_vals
  [ilq_traj, values] = run_ilqgames("one_player_reachability_example", "_feedback", ...
                                    a, nominal_control_penalty, x0_flag);
  plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', 'color', colormap(a, scale_vals, true), ...
       'DisplayName', sprintf(char("$a = %1.2f, " + value_format_string), a, ...
                              values(1) + value - 1.0));
end

hold off;
l1 = legend('Location', 'SouthWest');

figure;
title(sprintf('Sensitivity to Control Penalty ($a = %1.2f$)', nominal_scale), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
if (baseline)
  plot(traj(:, 1), traj(:, 2), 'g-o', 'DisplayName', 'Best-effort solution');
end

for epsilon = control_penalty_vals
  [ilq_traj, values] = run_ilqgames("one_player_reachability_example", "_feedback", ...
                                    nominal_scale, epsilon, x0_flag);
  plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', 'color', ...
       colormap(epsilon, control_penalty_vals, false), 'DisplayName', ...
       sprintf(char("$\\epsilon = %1.2f, " + value_format_string), epsilon, ...
               values(1) + value - 1.0));
end

hold off;
l2 = legend('Location', 'SouthWest');

set(l1, 'Interpreter', 'latex');
set(l2, 'Interpreter', 'latex');

end

function two_player_comparison(baseline)
% Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory

%% Grid
grid_min = [-1; -8; -pi; 0.0]; % Lower corner of computation domain
grid_max = [2; 1; pi; 2.0];    % Upper corner of computation domain
N = [23; 23; 11; 11];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);

%% Compute optimal trajectory from some initial state
%set the initial state
xinit = [0, -1.5, pi / 4, 0.5];

%% target set
R = 1.0;
data0 = shapeCylinder(g, [3, 4], [0; 0; 0; 0], R);

%% time vector
t0 = 0;
tMax = 2;
dt = 0.1;
tau = t0:dt:tMax;

if (baseline)
%% problem parameters

% input bounds
aMax = 0.1;
wMax = 1;
dMax = 0.5;

% control trying to min or max value function?
uMode = 'max';
dMode = 'min';

%% Pack problem parameters

% Define dynamic system
dCar = Plane4D([0; 0; 0; 0], wMax, [-aMax; aMax], [-dMax; dMax]); %do dStep3 here

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
%do dStep4 here

%% additive random noise
%do Step8 here
%HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
% Try other noise coefficients, like:
%    [0.2; 0; 0]; % Noise on X state
%    [0.2,0,0;0,0.2,0;0,0,0.5]; % Independent noise on all states
%    [0.2;0.2;0.5]; % Coupled noise on all states
%    {zeros(size(g.xs{1})); zeros(size(g.xs{1})); (g.xs{1}+g.xs{2})/20}; % State-dependent noise

%% If you have obstacles, compute them here

%% Compute value function

HJIextraArgs.visualize = false; %show plot
HJIextraArgs.fig_num = 1; %set figure number
HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'minVOverTime', HJIextraArgs);

%%check if this initial state is in the BRS/BRT
%value = eval_u(g, data, x)
value = eval_u(g,data(:,:,:,:,end),xinit);

dCar.x = xinit; %set initial state of the dubins car

TrajextraArgs.uMode = uMode; %set if control wants to min or max
TrajextraArgs.visualize = false; %show plot
TrajextraArgs.fig_num = 2; %figure number

%%we want to see the first two dimensions (x and y)
TrajextraArgs.projDim = [1 1 0 0];

%%flip data time points so we start from the beginning of time
dataTraj = flip(data,5);

[traj, traj_tau] = computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);
traj = traj'; % Transpose traj to have colums be different timesteps
value = eval_u(g, data(:,:,:,:,end), xinit);
end

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
scale_vals = linspace(1.0, 10.0, 5);
control_penalty_vals = linspace(1.0, 10.0, 5);

nominal_scale = 5.0;
nominal_control_penalty = 5.0;

figure;
title(sprintf('Sensitivity to Scale ($\\epsilon = %1.2f$)', nominal_control_penalty), ...
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


function collision_avoidance_example()
close all;

%% Compute ILQ trajectory for same problem with different parameters and overlay plots.
scale_vals = linspace(0.1, 0.5, 5);
control_penalty_vals = linspace(0.75, 1.5, 5);

nominal_scale = 0.5;
nominal_control_penalty = 0.75;

figure;
set(gca, 'FontSize', 24');
title(sprintf('Sensitivity to Scale ($\\epsilon = %1.2f$)', nominal_control_penalty), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
%%value_format_string = "\\tilde V(x_1) = %1.2f$";

buffer = 2.0;
x0_flag = "--d0=5.0 --v0=5.0 --buffer=" + buffer;

ii = 1;
for a = scale_vals
  [ilq_traj, values] = run_ilqgames("three_player_collision_avoidance_reachability_example", "", ...
                                    a, nominal_control_penalty, x0_flag);
  if max(abs(values(1) - values(2)), abs(values(1) - values(3))) > 0.01
    disp(char("Error! Values do not match: " + values(1) + ", " + values(2) + ", " + values(3)));
  end

  pa(ii) = plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', 'color', colormap(a, scale_vals, true), ...
                'DisplayName', sprintf('$a = %1.2f$', a));
  plot(ilq_traj(:, 6), ilq_traj(:, 7), 'x:', 'color', colormap(a, scale_vals, true));
  plot(ilq_traj(:, 11), ilq_traj(:, 12), 'x--', 'color', colormap(a, scale_vals, true));

  ii = ii + 1;
end

hold off;
l1 = legend(pa, 'Location', 'NorthEast');

figure;
set(gca, 'FontSize', 24');
title(sprintf('Sensitivity to Control Penalty ($a = %1.2f$)', nominal_scale), ...
      'Interpreter', 'latex');
xlabel('$p_x$ (m)', 'Interpreter', 'latex');
ylabel('$p_y$ (m)', 'Interpreter', 'latex');

hold on;
ii = 1;
for epsilon = control_penalty_vals
  [ilq_traj, values] = run_ilqgames("three_player_collision_avoidance_reachability_example", "", ...
                                    nominal_scale, epsilon, x0_flag);
  if max(abs(values(1) - values(2)), abs(values(1) - values(3))) > 0.01
    disp(char("Error! Values do not match: " + values(1) + ", " + values(2) + ", " + values(3)));
  end

  pe(ii) = plot(ilq_traj(:, 1), ilq_traj(:, 2), 'x-', ...
                'color', colormap(epsilon, control_penalty_vals, false), ...
                'DisplayName', sprintf('$\\epsilon = %1.2f$', epsilon));
  plot(ilq_traj(:, 6), ilq_traj(:, 7), 'x:', 'color', ...
       colormap(epsilon, control_penalty_vals, false));
  plot(ilq_traj(:, 11), ilq_traj(:, 12), 'x--', 'color', ...
       colormap(epsilon, control_penalty_vals, false));

  ii = ii + 1;
end

hold off;
l2 = legend(pe, 'Location', 'NorthEast');

set(l1, 'Interpreter', 'latex');
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
function [traj, values] = run_ilqgames(exec, extra_suffix, scale, control_penalty, x0_flag)
  experiment_name = exec + "_" + scale + "_" + control_penalty + x0_flag;
  experiment_arg = " --experiment_name='" + experiment_name + extra_suffix + "'";
  save_flag = "--save" + extra_suffix;

  if ~experiment_already_run(char(experiment_name + extra_suffix))
    %% Stitch together the command for the executable.
    instruction = "../bin/" + exec + " --noviz " + save_flag + ...
                  " --last_traj" + experiment_arg + " --exponential_constant=" + scale + ...
                  " --convergence_tolerance=0.01 --trust_region_size=0.1" + ...
                  " --control_penalty=" + control_penalty + ...
                  " --initial_alpha_scaling=0.01 " + x0_flag;
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
  for ii = 1:length(values)
    values(ii) = log(values(ii)) / scale;
  end
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
