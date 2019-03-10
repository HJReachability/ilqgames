function [traj, traj_tau] = runningSumComputeOptTraj(g, data, tau, dynSys, extraArgs)
% [traj, traj_tau] = computeOptTraj(g, data, tau, dynSys, extraArgs)
%   Computes the optimal trajectories given the optimal value function
%   represented by (g, data), associated time stamps tau, dynamics given in
%   dynSys.
%
% Inputs:
%   g, data - grid and value function
%   tau     - time stamp (must be the same length as size of last dimension of
%                         data)
%   dynSys  - dynamical system object for which the optimal path is to be
%             computed
%   extraArgs
%     .uMode        - specifies whether the control u aims to minimize or
%                     maximize the value function
%     .dMode        - specifies whether the disturbance d aims to minimize
%                     or maximize the value function (or 'none' if we do
%                     not want to simulate the disturbance)
%     .visualize    - set to true to visualize results
%     .fig_num:   List if you want to plot on a specific figure number
%     .projDim      - set the dimensions that should be projected away when
%                     visualizing
%     .fig_filename - specifies the file name for saving the visualizations

if nargin < 5
  extraArgs = [];
end

% Default parameters
uMode = 'min';
dMode = 'none';
visualize = false;
subSamples = 4;

if isfield(extraArgs, 'uMode')
  uMode = extraArgs.uMode;
end

if isfield(extraArgs, 'dMode')
  dMode = extraArgs.dMode;
end

% Visualization
if isfield(extraArgs, 'visualize') && extraArgs.visualize
  visualize = extraArgs.visualize;
  
  showDims = find(extraArgs.projDim);
  hideDims = ~extraArgs.projDim;
  
  if isfield(extraArgs,'fig_num')
    f = figure(extraArgs.fig_num);
  else
    f = figure;
  end
end

if isfield(extraArgs, 'subSamples')
  subSamples = extraArgs.subSamples;
end

clns = repmat({':'}, 1, g.dim);

if any(diff(tau)) < 0
  error('Time stamps must be in ascending order!')
end

% Time parameters
iter = 1;
tauLength = length(tau);

dtSmall = (tau(2) - tau(1))/subSamples;
% maxIter = 1.25*tauLength;

% Initialize trajectory
traj = nan(g.dim, tauLength);
traj(:,1) = dynSys.x;
tEarliest = 1;

while iter <= tauLength 
  fprintf('iter %d, length %d\n', iter, tauLength);
  % Determine the earliest time that the current state is in the reachable set
  % Binary search
  upper = tauLength;

  lower = tEarliest;
  
%   tEarliest = find_earliest_BRS_ind(g, data, dynSys.x, upper, lower);

  % BRS at current time
%   BRS_at_t = data(clns{:}, tEarliest);
%   BRS_at_t = data(clns{:}, end);
    BRS_at_t = data(:, :, :, :, 1);
  
  if isnan(eval_u(g, BRS_at_t, dynSys.x))
      fprintf('Value function at t = %f not defined!\n', tEarliest);
      break;
  end
  
  % Visualize BRS corresponding to current trajectory point
  if visualize
    plot(traj(showDims(1), iter), traj(showDims(2), iter), 'k.')
    hold on
    [g2D, data2D] = proj(g, BRS_at_t, hideDims, traj(hideDims, iter));
    visSetIm(g2D, data2D);
    tStr = sprintf('t = %.3f; tEarliest = %.3f', tau(iter), tau(tEarliest));
    title(tStr)
    drawnow
    
    if isfield(extraArgs, 'fig_filename')
      export_fig(sprintf('%s%d', extraArgs.fig_filename, iter), '-png')
    end

    hold off
  end
  
%   if tEarliest == tauLength
%     % Trajectory has entered the target
%     fprintf('Terminating because trajectory has entered the target\n');
%     break
%   end
  
  % Update trajectory
  Deriv = computeGradients(g, BRS_at_t);
  for j = 1:subSamples
    deriv = eval_u(g, Deriv, dynSys.x);
    if ~isfield(extraArgs, 'optCtrl')
        u = dynSys.optCtrl(tau(tEarliest), dynSys.x, deriv, uMode);
    else
        u = extraArgs.optCtrl(dynSys, deriv, extraArgs.R_u, uMode);
    end
    
    d = [];
    if ~strcmp(dMode, 'none')
      if ~isfield(extraArgs, 'optDist')
        d = dynSys.optDstb(tau(tEarliest), dynSys.x, deriv, dMode);
      else
        d = extraArgs.optDist(dynSys, deriv, extraArgs.R_d, dMode);
      end
    end
    
    dynSys.updateState(u, dtSmall, dynSys.x, d);
    if isnan(dynSys.x(1))
        fprintf('nan!\n');
    end
  end
  
  % Record new point on nominal trajectory
  iter = iter + 1;
  traj(:,iter) = dynSys.x;
end

% Delete unused indices
traj(:,iter:end) = [];
traj_tau = tau(1:iter-1);
end