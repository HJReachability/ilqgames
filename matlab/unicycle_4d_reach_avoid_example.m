clear all;

%% Parameters.
initState = [10; 10; pi/2.5; 5];

wMax = 1;
aMax = 2;
aRange = [-aMax; aMax];
dMax = [0; 0];
dynamics = Plane4D(initState, wMax, aRange, dMax);

%% Target and obstacles.
% gridCells = [25; 25; 25; 25];
% gridCells = [21; 21; 21; 21];
gridCells = [15; 15; 15; 15];
%gridCells = [40; 40; 40; 40];
periodicDim = 3;

g = createGrid([0; 0; -pi; -10], [175; 175; pi; 20], gridCells, periodicDim);

% Create the goal.
goalPos = [125, 100];
goalCost = ProximityCost([1, 2], goalPos, Inf, 0.01);
goalCostWeight = -1;

target = zeros(gridCells');

obstacleCenters = [100, 65, 25; 35, 65, 80];
obstacleRadii = [10, 10, 10];
obstacleCostWeights = [100, 100, 100];

% Control and disturbance quadratic cost weights. 
cost_u = 10;
cost_d = 10;

R_u = eye(2) * cost_u;
R_d = eye(2) * cost_d;

%% Compute reachable set
%tau = 0:0.5:500;
tau = 0:0.5:40;
% tau = 0:0.5:80;

uMode = 'min';
dMode = 'max';

minWith = 'none';

% For FRT, we set minWith to zero.
%minWith = 'zero';

% Temporary removed so that HJI PDE does not do the default computation
schemeData.dynSys = dynamics;
schemeData.grid = g;
schemeData.uMode = uMode;
schemeData.dMode = dMode;

% Add the state-dependent cost functions.
schemeData.stateCosts = {goalCost};
schemeData.stateCostWeights = {goalCostWeight};

for i = 1:length(obstacleRadii)
    schemeData.stateCosts{i+1} = ObstacleCost(...
        [1, 2], obstacleCenters(:, i)', obstacleRadii(i));
    schemeData.stateCostWeights{i+1} = obstacleCostWeights(i);
end

maxVel = 15;

schemeData.stateCosts{end+1} = SemiquadraticCost(4, maxVel, true);
schemeData.stateCostWeights{end+1} = 20;

schemeData.stateCosts{end+1} = SemiquadraticCost(4, 0, false);
schemeData.stateCostWeights{end+1} = 20;

schemeData.hamFunc = @runningSumUnicycle4DHam;
schemeData.partialFunc = @runningSumUnicycle4DPartial;
schemeData.R_u = R_u;
schemeData.R_d = R_d;
%schemeData.tMode = 'forward';
schemeData.tMode = 'backward';

% extraArgs.targets = target;
% extraArgs.obstacles = obs;
extraArgs.stopConverge = true;
% extraArgs.stopInit = dynamics.x;
extraArgs.visualize = true;
extraArgs.plotData.plotDims = [1 1 0 0];
extraArgs.plotData.projpt = dynamics.x(3:4);
extraArgs.deleteLastPlot = true;

if isfield(schemeData, 'hamFunc')
    data_filename = [mfilename '_wMax_' num2str(wMax) '_aMax_' ...
        num2str(aRange(2)) '_dMax_' num2str(dMax(2)) '_cost_u_' ...
        num2str(cost_u) '_cost_d_' num2str(cost_d) '.mat'];
else
    data_filename = [mfilename '_wMax_' num2str(wMax) '_aMax_' ...
        num2str(aRange(2)) '_dMax_' num2str(dMax(2)) '.mat'];
end

if exist(data_filename, 'file')
    load(data_filename);
else
    [data, tau2] = runningSumHJIPDE_solve(target, tau, schemeData, minWith, ...
        extraArgs);
    save(data_filename, 'data', 'tau2', 'g');
end

%% Compute optimal trajectory
extraArgs.projDim = [1 1 0 0]; 
% [traj, traj_tau] = computeOptTraj(g, flip(data,5), tau2, dynamics, extraArgs);

compTraj = true;
if compTraj
%   pause
  
  %set the initial state
%   xinit = [3, 3, -pi];
  
%   figure(6)
%   clf
%   h = visSetIm(g, data(:, :, :, :, end));
%   h.FaceAlpha = .3;
%   hold on
% %   s = scatter3(xinit(1), xinit(2), xinit(3));
%   s = scatter3(initState(1), initState(2), initState(3));
%   s.SizeData = 70;
  
  %check if this initial state is in the BRS/BRT
  value = eval_u(g, data(:, :, :, :, end), initState);
  
  % (HACK) Change bound for running cost
  if isfield(schemeData, 'hamFunc')
%     init_pt_upper_V = 200;
    init_pt_upper_V = Inf;
  else
    init_pt_upper_V = 0;
  end
  
  if value <= init_pt_upper_V %if initial state is in BRS/BRT
    % find optimal trajectory
    
    dynamics.x = initState; %set initial state of the dubins car

    trajExtraArgs.uMode = uMode; %set if control wants to min or max
    trajExtraArgs.dMode = dMode;
    trajExtraArgs.visualize = true; %show plot
    trajExtraArgs.fig_num = 2; %figure number
    
    %we want to see the first two dimensions (x and y)
    trajExtraArgs.projDim = [1 1 0 0]; 
    
    %flip data time points so we start from the beginning of time
    dataTraj = flip(data, 5);
    
    % Add the optimal control and disturbance function
    trajExtraArgs.optCtrl = @runningSumUnicycle4DOptCtrl;
    trajExtraArgs.optDist = @runningSumUnicycle4DOptDist;
    trajExtraArgs.R_u = R_u;
    trajExtraArgs.R_d = R_d;
    
    % Set the duration of the trajectory to be 10 s.
%     trajExtraArgs.duration = 10;
%     trajExtraArgs.timeStep = 0.01;
    
    % Compute the optimal trajectory (with distrubance).
    [traj, traj_tau] = ...
      runningSumComputeOptTraj(g, dataTraj, tau2, dynamics, trajExtraArgs);

    % Compute the optimal trajectory (with no disturbance).
    dynamics.x = initState;
    trajExtraArgs.dMode = 'none';
    
%     [traj_no_d, traj_tau_no_d] = ...
%       runningSumComputeOptTraj(g, dataTraj, tau2, dynamics, trajExtraArgs);
  
    hold on;
    
    plot(traj(1, :), traj(2, :), 'DisplayName', 'w/ dstb');
%     plot(traj_no_d(1, :), traj_no_d(2, :), 'DisplayName', 'w/o dstb');
    scatter(goalPos(1), goalPos(2), 'LineWidth', 3);
    xlim([0 150]);
    ylim([0 150]);
    legend();
    
    for ii = 1:size(obstacleRadii, 2)
       plotCircle(obstacleCenters(:, ii), obstacleRadii(ii), 'obs'); 
    end
    
    save('unicycle_4d_example_hji.mat', 'traj');
    
  else
    error(['Initial state is not in the BRS/BRT! It have a value of ' num2str(value, 2)])
  end
end

function plotCircle(center, radius, name)
th = 0:pi/50:2*pi;
xs = radius*cos(th) + center(1);
ys = radius*sin(th) + center(2);
plot(xs, ys, 'DisplayName', name);
end