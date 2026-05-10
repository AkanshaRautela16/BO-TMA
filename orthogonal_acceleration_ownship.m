%% Iterative TMA Attainable Set with Orthogonal Own-Ship Acceleration
clear; clc; close all;

% 1. Simulation Parameters
N_iter = 5;         % Number of measurement iterations
T = 1;             % Time step between measurements (s)
a_max = 0.0;        % Maximum target acceleration (m/s^2)
a_own_max = 0.05;    % Maximum own-ship acceleration (m/s^2)
R0 = 2000; V0 = 1500; % Initial uncertainty bounds

% Initial Relative State Estimate: X = [rx; vx; ry; vy]
mu_current = [0; 0; 0; 0]; 
P_current = blkdiag([R0^2 0; 0 V0^2], [R0^2 0; 0 V0^2]);

% True Relative Target Trajectory 
X_true = [500;58;700;-10]; 

% 2. Dynamics and Reachability Gramian
Phi_2D = [1 T; 0 1];
G_2D   = [ (1/3)*T^3, (1/2)*T^2; (1/2)*T^2, T ];
Gamma_2D = [0.5*T^2; T]; % Input mapping matrix

Phi_4D = blkdiag(Phi_2D, Phi_2D);
G_4D   = 0*(a_own_max^2) * blkdiag(G_2D, G_2D);

% Gamma_4D maps 2D accelerations [ax; ay] into the 4D state vector
Gamma_4D = [Gamma_2D(1) 0; 
            Gamma_2D(2) 0; 
            0 Gamma_2D(1); 
            0 Gamma_2D(2)];

% --- Setup Arrays for Error Tracking ---
t_vec = zeros(1, N_iter);
err_pos = zeros(1, N_iter);
err_vel = zeros(1, N_iter);
mu_current_vec =[];
X_true_vec = [];
% Setup Figure and Visualization parameters
figure('Color', 'w', 'Position', [100 100 1200 800]);
colors = lines(N_iter); % Colormap for iteration stages
theta = linspace(0, 2*pi, 100);
circle = [cos(theta); sin(theta)];

disp('Starting Iterative Filter...');

% 3. The Iterative Predict-Update Loop
for k = 1:N_iter
    
    % --- A. Generate Accelerations ---
    % Calculate current bearing to determine the orthogonal direction
    current_beta = atan2(X_true(1), X_true(3));
    
    % Generate a random acceleration magnitude
    accel_mag = a_own_max; 
    
    % Apply magnitude to the orthogonal vector [cos(beta); -sin(beta)]
    a_own = accel_mag * [cos(current_beta); -sin(current_beta)];
    
    % The true target applies a slight constant maneuver
    a_tgt = [0.0; -0.0]; 
    
    % --- B. Simulate True Relative Target ---
    % Relative state shifts by (Target Accel - Ownship Accel)
    X_true = Phi_4D * X_true + Gamma_4D * a_tgt - Gamma_4D * a_own;
    X_true_vec = [X_true_vec X_true]
    
    % Sensor takes a precise bearing measurement of the true relative target
    beta = atan2(X_true(1), X_true(3)); 
    H = [cos(beta), 0, -sin(beta), 0];
    
    % --- C. PREDICTION (Time Update) ---
    % The observer KNOWS its own acceleration, so it deterministically subtracts it
    % from the predicted mean without adding any new uncertainty.
    mu_pred = Phi_4D * mu_current - Gamma_4D * a_own;
    
    % Target's unknown acceleration adds uncertainty via G_4D
    P_pred  = Phi_4D * P_current * Phi_4D' + G_4D;
    
    % --- D. MEASUREMENT UPDATE (Slicing) ---
    % Calculate geometry of the cut
    S = H * P_pred * H';
    d2 = (H * mu_pred)^2 / S;
    
    if d2 > 1
        error('Bearing at step %d does not intersect the attainable set.', k);
    end
    
    % Extract the center and shape of the 3D slice
    mu_current = mu_pred - P_pred * H' * (1/S) * (H * mu_pred);
    P_c = P_pred - P_pred * H' * (1/S) * H * P_pred;
    mu_current_vec = [mu_current_vec mu_current]
    % CRITICAL: Scale the new covariance to account for off-center cuts
    scale_sq = 1 - d2; 
    P_current = P_c * scale_sq; 
    
    fprintf('Iteration %d: Orthogonal Own-Ship Accel = [%.2f, %.2f], Mahalanobis d^2 = %.3f\n', k, a_own(1), a_own(2), d2);
    
    % --- E. ERROR TRACKING ---
    t_vec(k) = k * T;
    err_pos(k) = norm(X_true([1,3]) - mu_current([1,3]));
    err_vel(k) = norm(X_true([2,4]) - mu_current([2,4]));
    
    % --- F. VISUALIZATION ---
    plot(mu_current([1]),mu_current([3]),'*')
    plot_projection(1, mu_pred, P_pred, mu_current, P_current, [1, 2], circle, colors(k,:), k, T);
    plot_projection(2, mu_pred, P_pred, mu_current, P_current, [3, 4], circle, colors(k,:), k, T);
    
    % Position Plane (Relative coordinates, sensor is always at origin)
    subplot(2,2,3); hold on; grid on;
    plot_projection(3, mu_pred, P_pred, mu_current, P_current, [1, 3], circle, colors(k,:), k, T);
    
    r_draw = 4000;
    plot([0, r_draw*sin(beta)], [0, r_draw*cos(beta)], '-.', 'Color', colors(k,:), 'LineWidth', 1.5);
    
    plot_projection(4, mu_pred, P_pred, mu_current, P_current, [2, 4], circle, colors(k,:), k, T);
end

% Formatting final phase plane plots
subplot(2,2,1); title('X-Phase Plane (rx vs vx)'); xlabel('rx (m)'); ylabel('vx (m/s)');
subplot(2,2,2); title('Y-Phase Plane (ry vs vy)'); xlabel('ry (m)'); ylabel('vy (m/s)');
subplot(2,2,3); title('Position Plane (rx vs ry)'); xlabel('rx (m)'); ylabel('ry (m)'); axis equal;
subplot(2,2,4); title('Velocity Plane (vx vs vy)'); xlabel('vx (m/s)'); ylabel('vy (m/s)'); axis equal;
sgtitle('Iterative Attainable Set: Orthogonal Own-Ship Maneuvers');

% --- G. PLOT ESTIMATION ERROR VS TIME ---
figure('Color', 'w', 'Position', [150 150 700 450]);

yyaxis left;
plot(t_vec, err_pos, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'auto');
ylabel('Relative Position Error (m)', 'FontWeight', 'bold');
ylim([0, max(err_pos)*1.2]);

yyaxis right;
plot(t_vec, err_vel, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'auto');
ylabel('Relative Velocity Error (m/s)', 'FontWeight', 'bold');
ylim([0, max(err_vel)*1.2]);

title('Target Tracking Estimation Error vs. Time', 'FontSize', 12);
xlabel('Time (s)', 'FontWeight', 'bold');
grid on;
xlim([0 t_vec(end) + T/2]);
ax = gca; ax.YAxis(1).Color = [0 0.4470 0.7410]; ax.YAxis(2).Color = [0.8500 0.3250 0.0980];

figure
plot(mu_current_vec(1,:),mu_current_vec(3,:),'r')
hold on
plot(X_true_vec(1,:),X_true_vec(3,:),'b')
%% Helper Function for Projecting, Plotting, and Labeling 2D Ellipses
function plot_projection(sub_idx, mu_pred, P_pred, mu_upd, P_upd, dims, circle, color, iter, T)
    subplot(2,2,sub_idx); hold on;
    
    % 1. Plot the Predicted (Inflated) Set as a dashed line
    P_proj_pred = P_pred(dims, dims);
    [V_p, D_p] = eig((P_proj_pred + P_proj_pred')/2);
    ell_pred = repmat(mu_pred(dims), 1, 100) + V_p * sqrt(max(D_p,0)) * circle;
    plot(ell_pred(1,:), ell_pred(2,:), '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
    
    % 2. Plot the Updated (Sliced) Set as a solid, filled region
    P_proj_upd = P_upd(dims, dims);
    [V_u, D_u] = eig((P_proj_upd + P_proj_upd')/2);
    ell_upd = repmat(mu_upd(dims), 1, 100) + V_u * sqrt(max(D_u,0)) * circle;
    
    fill(ell_upd(1,:), ell_upd(2,:), color, 'FaceAlpha', 0.15, 'EdgeColor', color, ...
         'LineWidth', 2, 'HandleVisibility', 'off');
         
    % 3. Label the time step
    text(mu_upd(dims(1)), mu_upd(dims(2)), sprintf(' t=%d', iter*T), ...
         'Color', color*0.8, 'FontSize', 10, 'FontWeight', 'bold');
end
