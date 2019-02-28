function [g,data,tau2]=plane4D_reachavoid()
% 1. Run Backward Reachable Set (BRS) with a goal
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = false <-- no trajectory
% 2. Run BRS with goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = true <-- compute optimal trajectory
% 3. Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory
% 4. Add disturbance
%     dStep1: define a dMax (dMax = [.25, .25, 0];)
%     dStep2: define a dMode (opposite of uMode)
%     dStep3: input dMax when creating your DubinsCar
%     dStep4: add dMode to schemeData
% 5. Change to an avoid BRT rather than a goal BRT
%     uMode = 'max' <-- avoid
%     dMode = 'min' <-- opposite of uMode
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = false <-- no trajectory
% 6. Change to a Forward Reachable Tube (FRT)
%     add schemeData.tMode = 'forward'
%     note: now having uMode = 'max' essentially says "see how far I can
%     reach"
% 7. Add obstacles
%     add the following code:
%     obstacles = shapeCylinder(g, 3, [-1.5; 1.5; 0], 0.75);
%     HJIextraArgs.obstacles = obstacles;

%% Should we compute the trajectory?
compTraj = 0;

%% Grid
grid_min = [-3.5; -3.5; -pi; -2]; % Lower corner of computation domain
grid_max = [3.5; 3.5; pi; 12];    % Upper corner of computation domain
N = [31; 31; 31; 31];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 1;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
%data0 = shapeCylinder(g, 3, [0; 4; 0], R);
data0 = shapeRectangleByCorners(g, [-1,-1, -inf, -inf], [1 1, inf, inf]);
% also try shapeRectangleByCorners, shapeSphere, etc.

%% time vector
t0 = 0;
tMax =2;
dt = 0.05;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
aRange = [0 1];
wMax = 1;
dMax = [0; 0];
%dMax = [.25 .25];
% do dStep1 here

% control trying to min or max value function?
uMode = 'min';
dMode = 'max';
% do dStep2 here


%% Pack problem parameters

% Define dynamic system
% obj = DubinsCar(x, wMax, speed, dMax)
%dCar = DubinsCar([0, 0, 0], wMax, speed); %do dStep3 here
dCar = Plane4D([0,0,0,0], wMax, aRange, dMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'low'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;
%do dStep4 here


%% If you have obstacles, compute them here
obstacle1 = shapeRectangleByCorners(g, [-3, -1, -inf, -inf], [-1, 1, inf inf]);
obstacle2 = -shapeRectangleByCorners(g, [-3, -3, -pi, 0],[3, 3, pi, 10]);

obstacle3 = shapeCylinder(g,[3,4],[(1+.5/sqrt(2)),(1+.5/sqrt(2))],.5);
obstacle4 = shapeCylinder(g,[3,4],[(1+.5/sqrt(2)),-(1+.5/sqrt(2))],.5);
obstacles = shapeUnion(obstacle1, obstacle2);
obstacles = shapeUnion(obstacles, obstacle3);
obstacles = shapeUnion(obstacles, obstacle4);
%obstacles = shapeCylinder(g, 3, [0; 0; 0], R);


%% Compute value function

HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
HJIextraArgs.obstacles = obstacles;

HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.plotData.plotDims = [1 1 0 0];
HJIextraArgs.visualize.plotData.projpt = [0,2];%'min';
HJIextraArgs.visualize.viewAngle = [0,90];

HJIextraArgs.stopConverge = 1;   
HJIextraArgs.keepLast = 1;

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'zero', HJIextraArgs);

%% Compute optimal trajectory from some initial state
if compTraj
  pause
  
  %set the initial state
  xinit = [3, 3, -pi];
  
  figure(6)
  clf
  h = visSetIm(g, data(:,:,:,end));
  h.FaceAlpha = .3;
  hold on
  s = scatter3(xinit(1), xinit(2), xinit(3));
  s.SizeData = 70;
  
  %check if this initial state is in the BRS/BRT
  %value = eval_u(g, data, x)
  value = eval_u(g,data(:,:,:,end),xinit);
  
  if value <= 0 %if initial state is in BRS/BRT
    % find optimal trajectory
    
    dCar.x = xinit; %set initial state of the dubins car

    TrajextraArgs.uMode = uMode; %set if control wants to min or max
    TrajextraArgs.visualize = true; %show plot
    TrajextraArgs.fig_num = 2; %figure number
    
    %we want to see the first two dimensions (x and y)
    TrajextraArgs.projDim = [1 1 0]; 
    
    %flip data time points so we start from the beginning of time
    dataTraj = flip(data,4);
    
    % [traj, traj_tau] = ...
    % computeOptTraj(g, data, tau, dynSys, extraArgs)
    [traj, traj_tau] = ...
      computeOptTraj(g, dataTraj, tau2, dCar, TrajextraArgs);
  else
    error(['Initial state is not in the BRS/BRT! It have a value of ' num2str(value,2)])
  end
end
end