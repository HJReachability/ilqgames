function d = runningSumUnicycle4DOptDist(dynSys, deriv, R_d, dMode)
%% Solves for the optimal control d for the following Hamiltonian:
%
%   min_u max_d grad_V^T f(x, u, d) + u^TR_u u - d^TR_d d
%
%  Arguments:
%      dynSys: an object of type dynSys
%      deriv: a cell array consisting of 4-dimensional arrays
%      R_d: a 2x2 matrix specifying the penalty for the cost of control 
%             input
%% Compute optimal disturbance 
    
    if ~iscell(deriv)
        deriv = num2cell(deriv);
    end
    
    if nargin < 4
        dMode = 'max';
    end
    
    if R_d(1,1) == 0
        d1_opt = 1e20;
    else
        d1_opt = 1.0 / (2 * R_d(1,1));
    end
    
    if R_d(2,2) == 0
        d2_opt = 1e20;
    else
        d2_opt = 1.0 / (2 * R_d(2,2));
    end
    d1_opt = d1_opt * deriv{1};
    d2_opt = d2_opt * deriv{2};
    
%     if ~isinf(d1_opt)
%         d1_opt = d1_opt * deriv{1};
%     else
%         d1_opt = d1_opt * ones(size(deriv{1}));
%     end
%     
%     if ~isinf(d2_opt)
%         d2_opt = d2_opt * deriv{2};
%     else
%         d2_opt = d2_opt * ones(size(deriv{2}));
%     end
    
    if strcmp(dMode, 'max')
        d1_opt(d1_opt > dynSys.dMax(1)) = dynSys.dMax(1);
        d1_opt(d1_opt < -dynSys.dMax(1)) = -dynSys.dMax(1);
        
        d2_opt(d2_opt > dynSys.dMax(2)) = dynSys.dMax(2);
        d2_opt(d2_opt < -dynSys.dMax(2)) = -dynSys.dMax(2);
        
        d{1} = d1_opt;
        d{2} = d2_opt;
        
        if isnan(d{1})
            fprintf('nan in optDist\n');
        end
    else
        %d1_opt = 1.0 / (2 * R_d(1,1)) * deriv{1};
        mask = zeros(size(d1_opt));
        mask(d1_opt>=0) = -1.0;
        mask(d1_opt<0) = 1.0;
        d{1} = dynSys.dMax(1) * mask;
        
        %d2_opt = -1.0 / (2 * R_d(2,2)) * deriv{2};
        mask = zeros(size(d2_opt));
        mask(d2_opt>=0) = -1.0;
        mask(d2_opt<0) = 1.0;
        d{2} = dynSys.dMax(2) * mask;
    end

end