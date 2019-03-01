%% Parameters.
% initState = [0; 0; pi/4; 10];
% initState = [10; 10; pi/4; 10];
initState = [10; 10; pi/4; 0];
wMax = 1;
aRange = [-2; 2];
% dMax = [0.1; 0.1];
% dMax = [0.8; 0.8];
dMax = [1.9; 1.9];
% dMax = [0; 0];
% dMax = [2; 2];
dynamics = Plane4D(initState, wMax, aRange, dMax);

%% Target and obstacles.
% gridCells = [25; 25; 25; 25];
% gridCells = [21; 21; 21; 21];
gridCells = [15; 15; 15; 15];
periodicDim = 3;

g = createGrid([0; 0; -pi; -1], [150; 150; pi; 30], gridCells, periodicDim);

targetRadius = 10;
targetCenter = [125; 100; 0; 0];
target = shapeCylinder(g, [3 4], targetCenter, targetRadius);

obstacleCenters = [40, 80, 100; 85, 110, 65];
obstacleRadii = [10, 10, 10];

obs1 = shapeCylinder(g, [3 4], obstacleCenters(:, 1), obstacleRadii(1));
obs2 = shapeCylinder(g, [3 4], obstacleCenters(:, 2), obstacleRadii(2));
obs3 = shapeCylinder(g, [3 4], obstacleCenters(:, 3), obstacleRadii(3));

obs = shapeUnion(obs1, obs2);
obs = shapeUnion(obs, obs3);

% obs = shapeRectangleByCorners( ...
%   g, [5; 5; -inf; -inf], [145; 145; inf; inf]);
% obs = -obs;

%% Compute reachable set
tau = 0:0.5:500;

uMode = 'min';
dMode = 'max';
minWith = 'none';

schemeData.dynSys = dynamics;
schemeData.grid = g;
schemeData.uMode = uMode;
schemeData.dMode = dMode;


extraArgs.targets = target;
extraArgs.obstacles = obs;
extraArgs.stopInit = dynamics.x;
extraArgs.visualize = true;
extraArgs.plotData.plotDims = [1 1 0 0];
extraArgs.plotData.projpt = dynamics.x(3:4);
extraArgs.deleteLastPlot = true;

[data, tau2] = HJIPDE_solve(target, tau, schemeData, minWith, extraArgs);

save(sprintf('%s.mat', mfilename), 'data', 'tau2', 'g');

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
  
  if value <= 0 %if initial state is in BRS/BRT
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
    
    % Set the duration of the trajectory to be 10 s.
%     trajExtraArgs.duration = 10;
%     trajExtraArgs.timeStep = 0.01;
    
    % Compute the optimal trajectory (with distrubance).
    [traj, traj_tau] = ...
      computeOptTraj(g, dataTraj, tau2, dynamics, trajExtraArgs);

    % Compute the optimal trajectory (with no disturbance).
    dynamics.x = initState;
    trajExtraArgs.dMode = 'none';
    
    [traj_no_d, traj_tau_no_d] = ...
      computeOptTraj(g, dataTraj, tau2, dynamics, trajExtraArgs);
  
    hold on;
    
    plot(traj(1, :), traj(2, :), 'DisplayName', 'w/ dstb');
    plot(traj_no_d(1, :), traj_no_d(2, :), 'DisplayName', 'w/o dstb');
    xlim([0 150]);
    ylim([0 150]);
    legend();
    
    for ii = 1:size(obstacleRadii, 2)
       plotCircle(obstacleCenters(:, ii), obstacleRadii(ii), 'obs'); 
    end
    
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