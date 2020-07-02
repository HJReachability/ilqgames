function hj_reachability_comparison()
% Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory

%% Should we compute the trajectory?
compTraj = true;

%% Grid
grid_min = [-5; -5; -pi]; % Lower corner of computation domain
grid_max = [5; 5; pi];    % Upper corner of computation domain
N = [41; 41; 41];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 0.5;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, 3, [0; 0; 0], R);
% also try shapeRectangleByCorners, shapeSphere, etc.

%% time vector
t0 = 0;
tMax = 2;
dt = 0.05;
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

%% Compute optimal trajectory from some initial state
if compTraj
  %set the initial state
  xinit = [2, 2, -pi];

  %check if this initial state is in the BRS/BRT
  %value = eval_u(g, data, x)
  value = eval_u(g,data(:,:,:,end),xinit);

  % find optimal trajectory
  dCar.x = xinit; %set initial state of the dubins car

  TrajextraArgs.uMode = uMode; %set if control wants to min or max
  TrajextraArgs.visualize = false; %show plot
  TrajextraArgs.fig_num = 2; %figure number

  %we want to see the first two dimensions (x and y)
  TrajextraArgs.projDim = [1 1 0];

  %flip data time points so we start from the beginning of time
  dataTraj = flip(data,4);

  % [traj, traj_tau] = ...
  % computeOptTraj(g, data, tau, dynSys, extraArgs)
  [traj, traj_tau] = computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);
  traj = traj'; % Transpose traj to have colums be different timesteps

  %% Compute ILQ trajectory for same problem.
  ilq_traj = run_ilqgames("one_player_reachability_example");
  if (size(traj, 1) ~= size(ilq_traj, 1))
    fprintf("Incorrect number of timesteps: %d vs. %d.", size(traj, 1), size(ilq_traj, 1));
  end

  if (size(traj, 2) ~= size(ilq_traj, 2))
    fprintf("Incorrect number of state dimensions: %d vs. %d.", size(traj, 2), size(ilq_traj, 2));
  end

  figure(3);
  hold on;
  plot(traj(:, 1), traj(:, 2), 'b-o');
  plot(ilq_traj(:, 1), ilq_traj(:, 2), 'g-o');
  hold off;
end
end

%% Compute ILQ trajectory for given example.
function traj = run_ilqgames(exec)
  experiment_name = "simple_avoid_feedback";
  experiment_arg = " --experiment_name='" + experiment_name + "'";

  exists = experiment_already_run(char(experiment_name));
  if ~exists
    %% Stitch together the command for the executable.
    instruction = "../bin/" + exec + " --noviz --save_feedback --last_traj" + ...
                  experiment_arg;
    system(char(instruction));
  end

  log_folder = "../logs/";
  cd(char(log_folder + experiment_name));
  dirs = dir;
  last_iterate = "blah";
  for ii = 1:size(dirs)
      if dirs(ii).name(1) ~= '.'
          last_iterate = dirs(ii).name;
          fprintf("%s", last_iterate);
          break;
      end
  end
  cd('../../matlab');
  traj = load(log_folder + experiment_name + "/" + last_iterate + "/xs.txt");
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
