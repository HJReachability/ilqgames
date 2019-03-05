function d = runningSumUnicycle4DOptDist(dynSys, deriv, R_d)
%% Solves for the optimal control d for the following Hamiltonian:
%
%   max_u min_d grad_V^T f(x, u, d) + u^TR_u u - d^TR_d d
%
%  Arguments:
%      dynSys: an object of type dynSys
%      deriv: a cell array consisting of 4-dimensional arrays
%      R_d: a 2x2 matrix specifying the penalty for the cost of control 
%             input
%% Compute optimal disturbance 
    d1_opt = -1.0 / (2 * R_d(1,1)) * deriv{1};
    mask = zeros(size(d1_opt));
    mask(d1_opt>=0) = -1.0;
    mask(d1_opt<0) = 1.0;
    d{1} = dynSys.dMax(1) * mask; 
    
    d2_opt = -1.0 / (2 * R_d(2,2)) * deriv{2};
    mask = zeros(size(d2_opt));
    mask(d2_opt>=0) = -1.0;
    mask(d2_opt<0) = 1.0;
    d{2} = dynSys.dMax(2) * mask; 

end