function alpha = runningSumUnicycle4DPartial(t, data, derivMin, ...
    derivMax, schemeData, dim)

g = schemeData.grid;
dynSys = schemeData.dynSys;

if ismethod(dynSys, 'partialFunc')
%   disp('Using partial function from dynamical system')
  alpha = dynSys.partialFunc(t, data, derivMin, derivMax, schemeData, dim);
  return
end

if ~isfield(schemeData, 'uMode')
  schemeData.uMode = 'min';
end

if ~isfield(schemeData, 'dMode')
  schemeData.dMode = 'min';
end

R_u = schemeData.R_u;
R_d = schemeData.R_d;
%% Compute control
if isfield(schemeData, 'uIn')
  % Control
  uU = schemeData.uIn;
  uL = schemeData.uIn;
 
else
    
  % Optimal control assuming maximum deriv
  uU = runningSumUnicycle4DOptCtrl(dynSys, derivMax, R_u);
  
  % Optimal control assuming minimum deriv
  uL = runningSumUnicycle4DOptCtrl(dynSys, derivMin, R_u);
end

%% Compute disturbance
if isfield(schemeData, 'dIn')
  dU = schemeData.dIn;
  dL = schemeData.dIn;
  
else
  dU = runningSumUnicycle4DOptDist(dynSys, derivMax, R_d);
  dL = runningSumUnicycle4DOptDist(dynSys, derivMin, R_d);
end
  
%% Compute alpha
dxUU = dynSys.dynamics(t, schemeData.grid.xs, uU, dU);
dxUL = dynSys.dynamics(t, schemeData.grid.xs, uU, dL);
dxLL = dynSys.dynamics(t, schemeData.grid.xs, uL, dL);
dxLU = dynSys.dynamics(t, schemeData.grid.xs, uL, dU);
alpha = max(abs(dxUU{dim}), abs(dxUL{dim}));
alpha = max(alpha, abs(dxLL{dim}));
alpha = max(alpha, abs(dxLU{dim}));
end
