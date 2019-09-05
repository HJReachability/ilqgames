%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for Control Cost 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% syms('vx','vy','v1','v2',"real")
% v = sqrt(vx*vx + vy*vy);
% cos_t = vx / v;
% sin_t = vy / v;
% M_inv = [cos_t sin_t; -sin_t/v cos_t/v];
% v_vec = [v1 ; v2];
% 
% MM = M_inv' * M_inv;
% 
% q_cost = v_vec'*(MM * v_vec);
% 
% dc_dvx = diff(q_cost, vx, 1)
% dc_dvy = diff(q_cost, vy, 1)
% 
% d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 1000)
% d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 1000)
% d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 1000)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for Control Cost 6D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% syms vx vy ax ay L w phi vn v1 v2
% v = sqrt(vx*vx + vy*vy);
% cos_t = vx / v;
% sin_t = vy / v;
% a = cos_t * ax + sin_t * ay;
% phi = atan((a * cos_t - ax) * L / (v * v * sin_t));
% cos_phi = cos(phi);
% tan_phi = tan(phi);
% 
% M_inv = [- v * v * sin_t /(cos_phi * cos_phi * L) cos_t;
%            v * v * cos_t /(cos_phi * cos_phi * L) sin_t];
%        
% m = [-a * sin_t * v * tan_phi/L - 2 * v/L * a * sin_t * tan_phi - v*v*v*tan_phi*tan_phi*cos_t/(L*L);
%       a * cos_t * v * tan_phi/L + 2 * v/L * a * cos_t * tan_phi - v*v*v*tan_phi*tan_phi*sin_t/(L*L)];
% 
% q_cost = (M_inv)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for orientation cost in terms of Xi{vx,vy}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

syms('vx','vy','w','tn',"real")
tan_inv = atan(vy/vx);

q_cost_t = 0.5 * w * (tan_inv - tn)^2;

dc_dvx = simplify(diff(q_cost_t, vx, 1))
dc_dvy = simplify(diff(q_cost_t, vy, 1))
 
d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 50)
d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 50)
d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 50)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for speed cost in terms of Xi{vx,vy}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% q_cost_v = 0.5 * w * (v - vn)^2;
% 
% dc_dvx = simplify(diff(q_cost_v, vx, 1), 'Steps', 50)
% dc_dvy = simplify(diff(q_cost_v, vy, 1), 'Steps', 50)
%  
% d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 50)
% d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 50)
% d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 50)
% 
% % Quadratic Cost Phi
% q_cost_phi = 0.5 * w * (phi - phin)^2
% 
% dc_dvx = simplify(diff(q_cost_phi, vx, 1), 'Steps', 50)
% dc_dvy = simplify(diff(q_cost_phi, vy, 1), 'Steps', 50)
% dc_dax = simplify(diff(q_cost_phi, ax, 1), 'Steps', 50)
% dc_day = simplify(diff(q_cost_phi, ay, 1), 'Steps', 50)
% 
% d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 50)
% d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 50)
% d2c_dvxdax = simplify(diff(dc_dvx, ax, 1), 'Steps', 50)
% d2c_dvxday = simplify(diff(dc_dvx, ay, 1), 'Steps', 50)
% 
% d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 50)
% d2c_dvydax = simplify(diff(dc_dvy, ax, 1), 'Steps', 50)
% d2c_dvyday = simplify(diff(dc_dvy, ay, 1), 'Steps', 50)
% 
% d2c_dax2 = simplify(diff(dc_dax, ax, 1), 'Steps', 50)
% d2c_daxday = simplify(diff(dc_dax, ay, 1), 'Steps', 50)
% 
% d2c_day2 = simplify(diff(dc_day, ay, 1), 'Steps', 50)