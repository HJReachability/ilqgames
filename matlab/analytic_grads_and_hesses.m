%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for Control Cost 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

syms('vx','vy','v1','v2',"real")
v = sqrt(vx*vx + vy*vy);
cos_t = vx / v;
sin_t = vy / v;
M_inv = [cos_t sin_t; -sin_t/v cos_t/v];
v_vec = [v1 ; v2];

Mv = M_inv * v_vec;

q_cost = Mv(2)^2; %Mv' * Mv;

dc_dvx = simplify(diff(q_cost, vx, 1))
dc_dvy = simplify(diff(q_cost, vy, 1))

d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 1000)
d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 1000)
d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 1000)

dc_dv1 = diff(q_cost, v1, 1);
dc_dv2 = diff(q_cost, v2, 1);

d2c_dv12 = simplify(diff(dc_dv1, v1, 1), 'Steps', 1000)
d2c_dv1dv2 = simplify(diff(dc_dv1, v2, 1), 'Steps', 1000)
d2c_dv22 = simplify(diff(dc_dv2, v2, 1), 'Steps', 1000)

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
% Gradient and Hessians for orientation cost in terms of x{theta}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% syms('t','tn','w',"real")
% 
% heading_x = cos(t);
% heading_y = sin(t);
% nom_headx = cos(tn);
% nom_heady = sin(tn);
% 
% inner_prod= heading_x * nom_headx + heading_y * nom_heady;
% angle_diff = acos(inner_prod);
% 
% q_cost_t = 0.5 * w * angle_diff^2;
% 
% dc_dt = simplify(diff(q_cost_t, t, 1),'Steps', 50)
% dc_dt2 = simplify(diff(dc_dt, t, 1), 'Steps', 50)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for orientation cost in terms of Xi{vx,vy}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% syms('vx','vy','w','h',"real")
% 
% Inv_rot = [cos(h),sin(h);-sin(h),cos(h)];
% vel_vec = [vx;vy];
% new_vec = Inv_rot * vel_vec;
% 
% angle_diff = atan(new_vec(2)/new_vec(1));
% 
% q_cost_t = 0.5 * w * angle_diff^2;
% 
% dc_dvx = simplify(diff(q_cost_t, vx, 1))
% dc_dvy = simplify(diff(q_cost_t, vy, 1))
%  
% d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 50)
% d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 50)
% d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 50)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient and Hessians for curvature cost in terms of Xi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% syms('vx','vy','ax','ay','L','w',"real")
% v = sqrt(vx*vx + vy*vy);
% cos_t = vx / v;
% sin_t = vy / v;
% a = cos_t * ax + sin_t * ay;
% phi = atan((a * cos_t - ax) * L / (v * v * sin_t));
% 
% q_cost_phi = 0.5 * w * phi^2;
% 
% dc_dvx = simplify(diff(q_cost_phi, vx, 1), 'Steps', 50)
% dc_dvy = simplify(diff(q_cost_phi, vy, 1), 'Steps', 50)
% dc_dax = simplify(diff(q_cost_phi, ax, 1), 'Steps', 50)
% dc_day = simplify(diff(q_cost_phi, ay, 1), 'Steps', 50)

%d2c_dvx2 = simplify(diff(dc_dvx, vx, 1), 'Steps', 50)
% d2c_dvxdvy = simplify(diff(dc_dvx, vy, 1), 'Steps', 50)
%d2c_dvxdax = simplify(diff(dc_dvx, ax, 1), 'Steps', 50)
%d2c_dvxday = simplify(diff(dc_dvx, ay, 1), 'Steps', 50)

% d2c_dvy2 = simplify(diff(dc_dvy, vy, 1), 'Steps', 50)
% d2c_dvydax = simplify(diff(dc_dvy, ax, 1), 'Steps', 50)
% d2c_dvyday = simplify(diff(dc_dvy, ay, 1), 'Steps', 50)
% 
% d2c_dax2 = simplify(diff(dc_dax, ax, 1), 'Steps', 50)
% d2c_daxday = simplify(diff(dc_dax, ay, 1), 'Steps', 50)
% 
% d2c_day2 = simplify(diff(dc_day, ay, 1), 'Steps', 50)

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