function u = runningSumUnicycle4DOptCtrl(dynSys, deriv, R_u)
%% Solves for the optimal control u for the following Hamiltonian:
%
%   max_u min_d grad_V^T f(x, u, d) + u^TR_u u - d^TR_d d
%
%  Arguments:
%      dynSys: an object of type dynSys
%      deriv: a cell array consisting of 4-dimensional arrays
%      R_u: a 2x2 matrix specifying the penalty for the cost of control 
%             input
%% Compute optimal control
    % u1 = w
    u1_opt = -1.0 / (2 * R_u(1,1)) * deriv{3};
    mask = zeros(size(u1_opt));
    mask(u1_opt>=0) = -1.0;
    mask(u1_opt<0) = 1.0;
    u{1} = dynSys.wMax * mask; 

    % u2 = a
    assert(dynSys.aRange(2) >= dynSys.aRange(1));
    u2_min = dynSys.aRange(1);
    u2_max = dynSys.aRange(2);
    
    H_u2_min = u2_min * deriv{4} + R_u(2,2) * u2_min * u2_min;
    H_u2_max = u2_max * deriv{4} + R_u(2,2) * u2_max * u2_max;
    u2 = ones(size(deriv{4})) * u2_min;
    u2(H_u2_max >= H_u2_min) = u2_max;

    u{2} = u2;
end