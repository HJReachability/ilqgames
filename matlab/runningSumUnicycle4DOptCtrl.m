function u = runningSumUnicycle4DOptCtrl(dynSys, deriv, R_u, uMode)
%% Solves for the optimal control u for the following Hamiltonian:
%
%   min_u max_d grad_V^T f(x, u, d) + u^TR_u u - d^TR_d d
%
%  Arguments:
%      dynSys: an object of type dynSys
%      deriv: a cell array consisting of 4-dimensional arrays
%      R_u: a 2x2 matrix specifying the penalty for the cost of control 
%             input
%% Compute optimal control

    if ~iscell(deriv)
        deriv = num2cell(deriv);
    end
    
    if nargin < 4
        uMode = 'min';
    end
    
    if R_u(1,1) == 0
        u1_opt = -1e20;
    else
        u1_opt = -1.0 / (2 * R_u(1,1));
    end
    
    if R_u(2,2) == 0
        u2_opt = -1e20;
    else
        u2_opt = -1.0 / (2 * R_u(2,2));
    end
    u1_opt = u1_opt * deriv{3};
    u2_opt = u2_opt * deriv{4};
%     u1_opt = -1.0 / (2 * R_u(1,1));
%     u2_opt = -1.0 / (2 * R_u(2,2));
%         
%     if ~isinf(u1_opt)
%         u1_opt = u1_opt * deriv{3};
%     else
%         if deriv{3} >= 0
%             u1_opt = u1_opt * ones(size(deriv{3}));
%         else
%             u1_opt = -u1_opt * ones(size(deriv{3}));
%         end
%     end
    
%     if ~isinf(u2_opt)
%         u2_opt = u2_opt * deriv{4};
%     else
%         if deriv{4} >= 0
%             u2_opt = u2_opt * ones(size(deriv{4}));
%         else
%             u2_opt = -u2_opt * ones(size(deriv{4}));
%         end
%     end
%     
    if strcmp(uMode, 'min')
        % u1 = w
        u1_opt(u1_opt > dynSys.wMax) = dynSys.wMax;
        u1_opt(u1_opt < -dynSys.wMax) = -dynSys.wMax;
        
        % u2 = a
        u2_opt(u2_opt > dynSys.aRange(2)) = dynSys.aRange(2);
        u2_opt(u2_opt < dynSys.aRange(1)) = dynSys.aRange(1);
    else
        mask = zeros(size(u1_opt));
        mask(u1_opt>=0) = -1.0;
        mask(u1_opt<0) = 1.0;
        u1_opt = dynSys.wMax * mask;
        
        % u2 = a
        assert(dynSys.aRange(2) >= dynSys.aRange(1));
        u2_min = dynSys.aRange(1);
        u2_max = dynSys.aRange(2);
        
        H_u2_min = u2_min * deriv{4} + R_u(2,2) * u2_min * u2_min;
        H_u2_max = u2_max * deriv{4} + R_u(2,2) * u2_max * u2_max;
        u2_opt = ones(size(deriv{4})) * u2_min;
        u2_opt(H_u2_max >= H_u2_min) = u2_max;    
    end
    u{1} = u1_opt;
    u{2} = u2_opt;
    dims = [1 2 3 4];
    u{1} = (deriv{dims==3}>=0)*(-dynSys.wMax) + ...
      (deriv{dims==3}<0)*dynSys.wMax;
  
    u{2} = (deriv{dims==4}>=0)*dynSys.aRange(1) + ...
      (deriv{dims==4}<0)*dynSys.aRange(2);

%     u_gt = dynSys.optCtrl(0, 0, deriv, uMode);
%     for i=1:2
%         if u_gt{i} == 0
%             printf('zeros');
%         end
%         if ~isequal(u{i},u_gt{i})
%             printf('not equal');
%         end
%     end
end