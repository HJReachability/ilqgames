function hamValue = runningSumUnicycle4DHam(t, data, deriv, schemeData)
%%
%  This functions compute the Hamiltonian for the following 4D unicycle 
%  dynamics and cost function
%
%  min_u max_d c(T) + \int{u^TR_u u - d^TR_d d}
%
%  which results in solving the following Hamiltonian
%
%  max_u min_d grad_V^T f(x, u, d) + u^TR_u u - d^TR_d d
%
%
%% Input unpacking
dynSys = schemeData.dynSys;

if ~isfield(schemeData, 'uMode')
  schemeData.uMode = 'min';
end

if ~isfield(schemeData, 'dMode')
  schemeData.dMode = 'min';
end

if ~isfield(schemeData, 'tMode')
  schemeData.tMode = 'backward';
end

%% Compute optimal control and disturbance
% Solve Hamiltonian based on dynamics and cost.
R_u = schemeData.R_u;
R_d = schemeData.R_d;

u = runningSumUnicycle4DOptCtrl(dynSys, deriv, R_u, schemeData.uMode);
d = runningSumUnicycle4DOptDist(dynSys, deriv, R_d, schemeData.dMode);

%% Plug optimal control into dynamics to compute Hamiltonian
hamValue = 0;
dx = dynSys.dynamics(t, schemeData.grid.xs, u, d);
for i = 1:dynSys.nx
  hamValue = hamValue + deriv{i}.*dx{i};
end

for i = 1:length(schemeData.stateCosts)
   weight = schemeData.stateCostWeights{i};
   hamValue = hamValue + weight * schemeData.stateCosts{i}.getCost(t, ...
       schemeData.grid.xs); 
end

hamValue = hamValue + R_u(1,1) * u{1}.^2 + R_u(2,2) * u{2}.^2;
hamValue = hamValue - R_d(1,1) * d{1}.^2 - R_d(2,2) * d{2}.^2;
hamValue = -hamValue;

end