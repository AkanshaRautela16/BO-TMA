%% smf_botma_paper.m
% Multi-sensor (bearing + Doppler) set-membership target motion analysis
% with optimal bounding ellipsoids, Grassmannian-adaptive own-ship
% maneuver, and a hard own-ship operating radius. Self-contained: runs
% directly in MATLAB (R2016b+) or GNU Octave (>= 6).
%
% [E1] Relative dynamics, x = target - own-ship = [rx vx ry vy]':
%        x_{k+1} = Phi x_k - Gamma a_k + Gamma w_k,   ||w_k||_2 <= aT
%      a_k = own-ship acceleration (known), w_k = target acceleration
%      (unknown, bounded). E(Q) = {Gamma w : ||w|| <= aT}, Q = aT^2 Gamma Gamma'.
% [E2] Measurements with bounded errors:
%        beta_k = atan2(rx,ry) + e_b,  |e_b| <= eps_b
%        z_k    = (rx vx + ry vy)/r + e_d,  |e_d| <= eps_d
% [E3] Bearing strip (exact): |H_b x| <= Delta_b, H_b = [cos b,0,-sin b,0],
%        Delta_b = rbar sin(eps_b), rbar = max range over the set.
% [E4] Doppler strip (first order + remainder gamma): |H_d x - c_d| <= eps_d + gamma
% [E5] Set E(mu,P) = {x : (x-mu)' P^-1 (x-mu) <= 1}; rank-one strip update:
%        mu+ = mu + rho P H' e / Om,  Om = 1 - rho + rho S,  S = H P H'
%        P+  = g/(1-rho) (P - rho P H' H P / Om)
%        g   = 1 - rho + rho Delta^2 - rho(1-rho) e^2 / Om
% [E6] Prediction, outer bound of the Minkowski sum (valid for any p>0):
%        P- = (1+1/p) Phi P Phi' + (1+p) Q,  p = argmin log det P-
%      (the plain sum Phi P Phi' + Q is NOT an outer bound: up to sqrt(2) short)
% [E7] Past-bearing strip carried forward, widened by target maneuver:
%        |H_b Phi^-1 x_k - c'| <= Delta_b + aT ||H_b Phi^-1 Gamma||
% [E8] Containment: q_k = (x_k-mu_k)' P_k^-1 (x_k-mu_k) <= 1 for all k.
% [E9] Grassmannian geometry: U_k = span([H_b; H_d]') in Gr(2,4),
%        d_Gr = ||theta||_2, cos(theta_i) = sigma_i(Q1'Q2); d_Gr ~ sqrt(2)|dbeta|.
%      Loiter speed adapted so windowed mean d_Gr tracks g_ref.
% [E10] Own-ship invariant: ||p|| + ||v||^2/(2 a_max) <= Rmax, enforced by
%       exact straight-line braking.
% [E11] Bearing-driven shuttle (prm.guidance = 'shuttle'): own-ship runs back
%       and forth across the measured LOS, perp = [cos b; -sin b], turning at
%       |perp'p| = Rleg. The cross-LOS baseline and the reversals (own-ship
%       acceleration across the LOS) are what make range observable.
%       Diagnostic: novelty angle between the new bearing row H_b,k and the
%       span of the last 3 bearing rows transported to time k (H_b,k-j Phi^-j);
%       ~0 means the bearing adds no new direction (range unobservable).
%
% Default: constant-velocity target, aT = 0 (classical TMA assumption:
% exact prediction). Validated numerically in GNU Octave 8.4 for N = 50..1000
% (0 containment violations). For a maneuvering target use
% targetMode = 'loiter' with aT >= speed^2/Rturn; its guaranteed accuracy
% floor is much larger (~280 m), see paper discussion.

clear; clc; close all;

%% ======================= USER PARAMETERS =======================
N          = 100;            % number of measurement steps (any N >= 2)
targetMode = 'cv';           % 'cv' | 'stationary' | 'loiter'
runCompare = true;           % Monte Carlo comparison over Ncompare
Ncompare   = [50 100 200 500];
nTrials    = 10;
seed       = 7;

%% ======================= MODEL PARAMETERS ======================
prm.T        = 4;               % sample period (s)
prm.epsBear  = deg2rad(0.5);    % [E2] bearing error bound
prm.epsDop   = 0.5;             % [E2] Doppler error bound (m/s)
prm.aT       = 0;               % [E1] target maneuver bound (m/s^2):
                                %   0 for 'cv'/'stationary' (exact prediction),
                                %   >= speed^2/Rturn (e.g. 0.016) for 'loiter'

% target truth (absolute frame; own-ship starts at the origin)
prm.tgt.p0    = [800; 1500];    % initial position (m)
prm.tgt.speed = 1.5;            % speed (m/s) for 'loiter' and 'cv'
prm.tgt.head  = deg2rad(-60);   % initial heading (rad, from +x toward +y)
prm.tgt.Rturn = 150;            % loiter turn radius (m): accel = speed^2/Rturn

% initial set [E5]
prm.Rinit = 1500;  prm.Rspan = 1000;  prm.Vspan = 5;
% Doppler remainder bound [E4]
prm.dopSamples = 300;  prm.dopSafety = 1.15;

% own-ship guidance [E9],[E10],[E11]
prm.guidance = 'shuttle';       % 'shuttle' (cross-LOS legs, [E11]) | 'loiter' (circle)
prm.Rleg    = 150;              % shuttle: turn around at this cross-LOS offset (m)
prm.Vleg    = 4;                % shuttle: leg speed (m/s)
prm.Rmax    = 200;              % hard operating radius around start (m)
prm.Rorb    = 95;               % loiter circle radius (circle passes through start)
prm.aMaxOwn = 0.5;              % own-ship acceleration limit (m/s^2)
prm.Vw0     = 4;  prm.VwMin = 1.5;
prm.VwMax   = min(6, sqrt(0.6*prm.aMaxOwn*prm.Rorb));  % centripetal <= 60% of a_max
prm.gRef    = 0.01;             % target windowed d_Gr (rad/step)
prm.kAdapt  = 0.05;  prm.win = 10;

% discrete kinematics [E1]
T = prm.T;
prm.Phi    = [1 T 0 0; 0 1 0 0; 0 0 1 T; 0 0 0 1];
prm.PhiInv = inv(prm.Phi);
prm.Gamma  = [T^2/2 0; T 0; 0 T^2/2; 0 T];
prm.Q      = prm.aT^2*(prm.Gamma*prm.Gamma');

%% ======================= SINGLE DETAILED RUN ===================
tgt = generateTarget(N, prm, targetMode);   % target truth, all N steps
out = runSMF(N, prm, tgt, seed);            % filter + guidance
t   = (0:N-1)*T;

fprintf('N = %d, target mode = %s, guidance = %s\n', N, targetMode, prm.guidance);
fprintf('  mean bearing novelty %.4f rad\n', mean(out.nov, 'omitnan'));
fprintf('  containment q_k <= 1 at %d/%d steps (max q = %.3f)\n', ...
    sum(out.q <= 1+1e-6), N, max(out.q));
fprintf('  own-ship max distance from start: %.1f m (limit %d m)\n', ...
    max(hypot(out.xOwn(1,:), out.xOwn(3,:))), prm.Rmax);
fprintf('  final position error %.1f m (bound %.1f m), velocity error %.3f m/s\n', ...
    out.posErr(end), out.posBound(end), out.velErr(end));

plotScenario(out, tgt, prm, t);
plotErrors(out, t);
plotSetSize(out, t);
plotGeometry(out, prm, t);
plotConvergence(out, t);

%% ======================= COMPARISON ACROSS N ===================
if runCompare
    compareN(Ncompare, nTrials, prm, targetMode);
end


%% ===================================================================
%% ======================= LOCAL FUNCTIONS ===========================
%% ===================================================================

function tgt = generateTarget(N, prm, mode)
% Target truth obeying [E1] exactly: x_{k+1} = Phi x_k + Gamma w_k,
% ||w_k|| <= aT (checked, so the scenario never violates the filter model).
T = prm.T;
x = zeros(4,N);  w = zeros(2,N);
switch lower(mode)
    case 'stationary', v0 = [0; 0];
    otherwise,         v0 = prm.tgt.speed*[cos(prm.tgt.head); sin(prm.tgt.head)];
end
x(:,1) = [prm.tgt.p0(1); v0(1); prm.tgt.p0(2); v0(2)];
if strcmpi(mode,'loiter')
    dth = prm.tgt.speed*T/prm.tgt.Rturn;
    Rot = [cos(dth) -sin(dth); sin(dth) cos(dth)];
end
for k = 1:N-1
    if strcmpi(mode,'loiter')
        vk = x([2 4],k);
        w(:,k) = (Rot*vk - vk)/T;
    end
    x(:,k+1) = prm.Phi*x(:,k) + prm.Gamma*w(:,k);
end
if max(sqrt(sum(w.^2,1))) > prm.aT + 1e-12
    error('Target maneuver %.4f m/s^2 exceeds model bound aT = %.4f; increase prm.aT.', ...
        max(sqrt(sum(w.^2,1))), prm.aT);
end
tgt.x = x;  tgt.w = w;  tgt.mode = mode;
end

function out = runSMF(N, prm, tgt, seed)
% Closed loop: measurements [E2] -> strip updates [E3]-[E5],[E7]
% -> Grassmannian adaptation [E9] -> guidance with invariant [E10] -> prediction [E6].
rng(seed);
T = prm.T;
xo = zeros(4,1);                    % own-ship absolute state (at rest at origin)

out.xRel = zeros(4,N);  out.xOwn = zeros(4,N);  out.mu = zeros(4,N);
out.P = zeros(4,4,N);
z = zeros(1,N);
[out.posErr, out.velErr, out.posBound, out.velBound, out.q, out.area, ...
 out.areaPrior, out.logVol, out.logVolPrior, out.semiAlong, out.semiCross, ...
 out.errAlong, out.errCross, out.Vw, out.rTrue, out.rEst] = deal(z);
out.dGr = nan(1,N);  out.dBeta = nan(1,N);  out.rank2 = nan(1,N);  out.rank3 = nan(1,N);
out.nov = nan(1,N);  out.fenceSteps = 0;

past = [];  aPrev = zeros(2,1);  Uhist = cell(1,3);  Vw = prm.Vw0;
center = zeros(2,1);  theta = 0;  betaPrev = NaN;  side = 1;  rows = zeros(0,4);
mu = zeros(4,1);  P = eye(4);

for k = 1:N
    % ---------- truth and measurements [E2] ----------
    x = tgt.x(:,k) - xo;
    out.xRel(:,k) = x;  out.xOwn(:,k) = xo;
    r = hypot(x(1),x(3));
    beta = atan2(x(1),x(3)) + (2*rand-1)*prm.epsBear;
    zdop = (x(1)*x(2) + x(3)*x(4))/r + (2*rand-1)*prm.epsDop;

    if k == 1
        % ---------- sensor-aligned initial set [E5] ----------
        % product of position and velocity ellipses contained in blkdiag(2Ppos,2Pvel)
        b0 = beta;  los0 = [sin(b0); cos(b0)];
        mu = [prm.Rinit*los0(1); zdop*los0(1); prm.Rinit*los0(2); zdop*los0(2)];
        Rr = [sin(b0) cos(b0); cos(b0) -sin(b0)];
        Dinit = (prm.Rinit + prm.Rspan)*sin(prm.epsBear);
        P = zeros(4);
        P([1 3],[1 3]) = Rr*diag([prm.Rspan^2, Dinit^2])*Rr';
        P([2 4],[2 4]) = Rr*diag([prm.epsDop^2, prm.Vspan^2])*Rr';
        P = regularizeEllipsoid(2*P);
        center = prm.Rorb*los0;  theta = atan2(-los0(2), -los0(1));
        GA = zeros(4,1);
        out.areaPrior(k) = ellArea(P);  out.logVolPrior(k) = logVol(P);
    else
        GA = prm.Gamma*aPrev;
    end

    % ---------- cut 1: previous bearing carried forward [E7] ----------
    if ~isempty(past)
        Hp = past.H*prm.PhiInv;
        Dp = past.Delta + prm.aT*norm(Hp*prm.Gamma);
        [mu,P] = rankOneCut(mu, P, Hp, past.c - Hp*GA, Dp);
    end
    % ---------- cut 2: current bearing, exact strip [E3] ----------
    Hb = [cos(beta) 0 -sin(beta) 0];
    rbar = hypot(mu(1),mu(3)) + sqrt(max(eig(P([1 3],[1 3]))));
    Db = rbar*sin(prm.epsBear);
    [mu,P] = rankOneCut(mu, P, Hb, 0, Db);
    past = struct('H',Hb,'c',0,'Delta',Db);
    % ---------- observability novelty of this bearing [E11] ----------
    if ~isempty(rows), rows = rows*prm.PhiInv; end       % transport past rows to time k
    if size(rows,1) >= 3
        [Qp,~] = qr(rows(end-2:end,:)',0);
        out.nov(k) = acos(min(1, norm(Qp'*(Hb'/norm(Hb)))));
    end
    rows = [rows(max(1,end-4):end,:); Hb];
    % ---------- cut 3: Doppler, linearised about current centre [E4] ----------
    [Hm, rc, h] = measJacobian(mu);
    Hd = Hm(2,:);
    gam = prm.dopSafety*dopplerRemainder(mu, P, Hd, h, prm.dopSamples);
    [mu,P] = rankOneCut(mu, P, Hd, zdop - h + Hd*mu, prm.epsDop + gam);

    % ---------- metrics [E8] ----------
    e = x - mu;
    out.mu(:,k) = mu;  out.P(:,:,k) = P;
    out.q(k) = (e'/P)*e;
    out.posErr(k) = hypot(e(1),e(3));  out.velErr(k) = hypot(e(2),e(4));
    out.posBound(k) = sqrt(max(eig(P([1 3],[1 3]))));
    out.velBound(k) = sqrt(max(eig(P([2 4],[2 4]))));
    out.area(k) = ellArea(P);  out.logVol(k) = logVol(P);
    losE = mu([1 3])/rc;  perpE = [losE(2); -losE(1)];
    out.semiAlong(k) = sqrt(losE'*P([1 3],[1 3])*losE);
    out.semiCross(k) = sqrt(perpE'*P([1 3],[1 3])*perpE);
    out.errAlong(k) = abs(losE'*e([1 3]));  out.errCross(k) = abs(perpE'*e([1 3]));
    out.rTrue(k) = r;  out.rEst(k) = rc;

    % ---------- Grassmannian measurement subspace [E9] ----------
    Hm = measJacobian(mu);
    [Uk,~] = qr(Hm',0);
    if ~isempty(Uhist{3}), out.dGr(k) = grassmannDistance(Uhist{3}, Uk); end
    if ~isnan(betaPrev), out.dBeta(k) = abs(angleWrap(atan2(mu(1),mu(3)) - betaPrev)); end
    betaPrev = atan2(mu(1),mu(3));
    Uhist = {Uhist{2}, Uhist{3}, Uk};
    if k >= 2, out.rank2(k) = rank([Uhist{2} Uhist{3}], 1e-8); end
    if k >= 3, out.rank3(k) = rank([Uhist{1} Uhist{2} Uhist{3}], 1e-8); end
    if k >= prm.win
        gbar = mean(out.dGr(k-prm.win+1:k), 'omitnan');
        Vw = min(max(Vw*(1 + prm.kAdapt*(prm.gRef - gbar)/prm.gRef), prm.VwMin), prm.VwMax);
    end
    out.Vw(k) = Vw;

    % ---------- own-ship guidance: loiter circle through the start ----------
    p = xo([1 3]);  vo = xo([2 4]);
    thetaNext = theta + Vw*T/prm.Rorb;
    pRef = center + prm.Rorb*[cos(thetaNext); sin(thetaNext)];
    vRef = Vw*[-sin(thetaNext); cos(thetaNext)];
    a = 0.5*(pRef - p - T*vo)/(0.5*T^2) + 0.5*(vRef - vo)/T;
    if strcmpi(prm.guidance, 'shuttle')
        % ---------- [E11] shuttle across the measured line of sight ----------
        losM  = [sin(beta); cos(beta)];
        perpM = [cos(beta); -sin(beta)];
        if side*(perpM'*p) >= prm.Rleg, side = -side; end        % end of leg: turn
        vCmd = side*prm.Vleg*perpM - 0.05*(losM'*p)*losM;        % hold the cross-LOS axis
        a = (vCmd - vo)/T;
        out.Vw(k) = prm.Vleg;
    end
    if norm(a) > prm.aMaxOwn, a = a*prm.aMaxOwn/norm(a); end
    % ---------- hard safety invariant [E10] ----------
    pn = p + T*vo + 0.5*T^2*a;  vn = vo + T*a;
    if norm(pn) + (vn'*vn)/(2*prm.aMaxOwn) > prm.Rmax
        sp = norm(vo);
        if sp > 1e-12, a = -min(prm.aMaxOwn, sp/T)*vo/sp; else, a = zeros(2,1); end
        out.fenceSteps = out.fenceSteps + 1;
    end
    theta = thetaNext;  aPrev = a;

    % ---------- propagate own-ship and set estimate [E1],[E6] ----------
    xo = prm.Phi*xo + prm.Gamma*a;
    if k < N
        mu = prm.Phi*mu - prm.Gamma*a;
        P  = minkowskiOuter(prm.Phi*P*prm.Phi', prm.Q);
        out.areaPrior(k+1) = ellArea(P);  out.logVolPrior(k+1) = logVol(P);
    end
end
end

function P = minkowskiOuter(A, Q)
% [E6] (1+1/p)A + (1+p)Q contains E(A)(+)E(Q) for every p>0 (AM-GM on the
% support functions); p chosen to minimise volume. Q = 0 gives P = A exactly.
if trace(Q) <= 0, P = A; return; end
f = @(lp) logdetSym((1+exp(-lp))*A + (1+exp(lp))*Q);
lp = fminbnd(f, -25, 10);
P = regularizeEllipsoid((1+exp(-lp))*A + (1+exp(lp))*Q);
end

function [muNew, PNew] = rankOneCut(mu0, P0, H, c, Delta)
% [E5] min-volume ellipsoid containing E(mu0,P0) intersected with |Hx-c|<=Delta.
e  = c - H*mu0;  S = max(H*P0*H', 0);
PH = P0*H';  PHH = PH*PH';
rho = fminbnd(@(r) cutLogDet(r, P0, PHH, S, e, Delta), 1e-9, 1-1e-9, optimset('TolX',1e-10));
Om = 1 - rho + rho*S;
g  = 1 - rho + rho*Delta^2 - rho*(1-rho)*e^2/Om;
muNew = mu0 + rho*PH*e/Om;
PNew  = regularizeEllipsoid((g/(1-rho))*(P0 - rho*PHH/Om));
if g <= 0 || any(~isfinite(muNew)) || any(~isfinite(PNew(:))) || min(eig(PNew)) <= 0
    muNew = mu0;  PNew = P0;          % not cutting is always a valid outer set
end
end

function ld = cutLogDet(rho, P0, PHH, S, e, Delta)
Om = 1 - rho + rho*S;
g  = 1 - rho + rho*Delta^2 - rho*(1-rho)*e^2/Om;
if g <= 0, ld = 1e300; return; end
ld = logdetSym((g/(1-rho))*(P0 - rho*PHH/Om));
end

function gamma = dopplerRemainder(mu, P, H, h, nS)
% [E4] numerical estimate of max over E(mu,P) of |h(x)-h(mu)-H(x-mu)|:
% 8 principal-axis endpoints + nS/2 boundary + nS/2 interior samples.
[V,D] = eig((P+P')/2);
L = V*diag(sqrt(max(diag(D),0)));
dAx = [eye(4), -eye(4)];
dRnd = randn(4,nS);  dRnd = dRnd./repmat(sqrt(sum(dRnd.^2,1)),4,1);
rad = [ones(1,8+floor(nS/2)), sqrt(rand(1,nS-floor(nS/2)))];
Dall = [dAx, dRnd];
X = repmat(mu,1,size(Dall,2)) + L*(Dall.*repmat(rad,4,1));
rs = hypot(X(1,:), X(3,:));  ok = rs > 1e-6;
if ~any(ok), gamma = 0; return; end
hT = (X(1,ok).*X(2,ok) + X(3,ok).*X(4,ok))./rs(ok);
gamma = max(abs(hT - (h + H*(X(:,ok) - repmat(mu,1,sum(ok))))));
end

function [H, r, h] = measJacobian(m)
r = hypot(m(1),m(3));  b = atan2(m(1),m(3));
h = (m(1)*m(2) + m(3)*m(4))/r;
H = [cos(b), 0, -sin(b), 0;
     m(2)/r - h*m(1)/r^2, m(1)/r, m(4)/r - h*m(3)/r^2, m(3)/r];
end

function d = grassmannDistance(Q1, Q2)
s = svd(Q1'*Q2);
d = norm(acos(min(max(s,-1),1)));
end

function a = angleWrap(a)
a = mod(a + pi, 2*pi) - pi;
end

function ld = logdetSym(M)
ev = eig((M+M')/2);
ld = sum(log(max(ev, realmin)));
end

function A = ellArea(P)
A = pi*sqrt(max(det(P([1 3],[1 3])), 0));
end

function lv = logVol(P)
lv = 0.5*logdetSym(P)/log(10);
end

function P = regularizeEllipsoid(P)
P = (P+P')/2;
[V,D] = eig(P);  D = diag(D);
D = max(D, max(D)*1e-9);
P = V*diag(D)*V';  P = (P+P')/2;
end

function cm = timeColormap(n)
try, cm = parula(n); catch, cm = jet(n); end
end

function plotEllipse(c, P2, col, alph)
[V,D] = eig((P2+P2')/2);
th = linspace(0, 2*pi, 60);
pts = V*diag(sqrt(max(diag(D),0)))*[cos(th); sin(th)] + repmat(c,1,60);
h = fill(pts(1,:), pts(2,:), col);
set(h, 'FaceAlpha', alph, 'EdgeColor', col, 'EdgeAlpha', min(1,3*alph));
end

function plotScenario(out, tgt, prm, t)
N = numel(t);
tgtTrue = tgt.x([1 3],:);
tgtEst  = out.xOwn([1 3],:) + out.mu([1 3],:);
cm = timeColormap(N);  stepE = max(1, floor(N/60));
figure('Name','Fig. 1  Scenario and tracking','Position',[30 40 1500 560]);
subplot(1,3,1); hold on;
for k = 1:stepE:N, plotEllipse(tgtEst(:,k), out.P([1 3],[1 3],k), cm(k,:), 0.06); end
plot(out.xOwn(1,:), out.xOwn(3,:), 'm-', 'LineWidth', 1.4);
plot(tgtTrue(1,:), tgtTrue(2,:), 'k-', 'LineWidth', 2);
plot(tgtEst(1,:), tgtEst(2,:), 'b.', 'MarkerSize', 4);
th = linspace(0,2*pi,200);
plot(prm.Rmax*cos(th), prm.Rmax*sin(th), 'r--');
plot(0,0,'m^','MarkerFaceColor','m');  plot(tgtTrue(1,1),tgtTrue(2,1),'ko','MarkerFaceColor','g');
axis equal; grid on; xlabel('East (m)'); ylabel('North (m)');
title('Scene: own-ship, target, set estimates');
legend('uncertainty ellipses (colour = time)','own-ship','true target','estimated centre', ...
       sprintf('%d m own-ship limit',prm.Rmax),'Location','southoutside');
subplot(1,3,2); hold on;
kz = max(1, round(N/4)):N;
for k = kz(1):stepE:N, plotEllipse(tgtEst(:,k), out.P([1 3],[1 3],k), cm(k,:), 0.08); end
plot(tgtTrue(1,:), tgtTrue(2,:), 'k-', 'LineWidth', 2);
plot(tgtEst(1,kz), tgtEst(2,kz), 'b.', 'MarkerSize', 6);
plot(tgtTrue(1,end), tgtTrue(2,end), 'ks', 'MarkerFaceColor','r');
axis equal; grid on; xlabel('East (m)'); ylabel('North (m)');
title(sprintf('Target tracking, t >= %.0f s (zoom)', t(kz(1))));
legend('ellipses','true trajectory','estimated centre','final true position','Location','southoutside');
subplot(1,3,3); hold on;
plot(out.xOwn(1,:), out.xOwn(3,:), 'm-', 'LineWidth', 1.4);
plot(prm.Rmax*cos(th), prm.Rmax*sin(th), 'r--', 'LineWidth', 1.2);
plot(0,0,'k^','MarkerFaceColor','k');
axis equal; grid on; xlabel('East (m)'); ylabel('North (m)');
title(sprintf('Own-ship trajectory (max %.1f m, fence active %d steps)', ...
    max(hypot(out.xOwn(1,:),out.xOwn(3,:))), out.fenceSteps));
legend('own-ship', sprintf('%d m limit',prm.Rmax), 'start', 'Location','southoutside');
end

function plotErrors(out, t)
figure('Name','Fig. 2  Estimation error and guarantee','Position',[40 60 1500 780]);
subplot(2,2,1);
semilogy(t, out.posErr, 'b-', 'LineWidth', 1.2); hold on;
semilogy(t, out.posBound, 'r--', 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('m');
title('Position error ||r_k - \mu_{r,k}|| and bound sqrt(\lambda_{max}(P_{rr}))');
legend('position error','guaranteed bound');
subplot(2,2,2);
semilogy(t, out.velErr, 'b-', 'LineWidth', 1.2); hold on;
semilogy(t, out.velBound, 'r--', 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('m/s');
title('Velocity error ||v_k - \mu_{v,k}|| and bound sqrt(\lambda_{max}(P_{vv}))');
legend('velocity error','guaranteed bound');
subplot(2,2,3);
semilogy(t, out.errAlong, 'b-', 'LineWidth', 1); hold on;
semilogy(t, out.semiAlong, 'b--', 'LineWidth', 1.4);
semilogy(t, out.errCross, 'g-', 'LineWidth', 1);
semilogy(t, out.semiCross, 'g--', 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('m');
title('Position error split: along-LOS (range) vs cross-LOS (bearing)');
legend('|error| along LOS','set extent along LOS','|error| cross LOS','set extent cross LOS');
subplot(2,2,4);
plot(t, out.q, 'b.-', 'MarkerSize', 6); hold on;
plot([t(1) t(end)], [1 1], 'r--', 'LineWidth', 1.5);
grid on; xlabel('Time (s)'); ylabel('q_k'); ylim([0 1.2]);
title(sprintf('Containment q_k = e_k^T P_k^{-1} e_k (<= 1 at %d/%d steps)', ...
    sum(out.q <= 1+1e-6), numel(t)));
legend('q_k','q = 1 (set boundary)');
end

function plotSetSize(out, t)
figure('Name','Fig. 3  Set-estimate size','Position',[50 80 1500 460]);
tt = reshape([t; t], 1, []);
subplot(1,3,1);
aa = reshape([out.areaPrior; out.area], 1, []);
semilogy(tt, aa, 'Color', [0.6 0.6 0.6]); hold on;
semilogy(t, out.area, 'k-', 'LineWidth', 1.5);
grid on; xlabel('Time (s)'); ylabel('m^2');
title('Position-ellipse area \pi sqrt(det P_{rr})');
legend('predict/update cycle','after update');
subplot(1,3,2);
vv = reshape([out.logVolPrior; out.logVol], 1, []);
plot(tt, vv, 'Color', [0.6 0.6 0.6]); hold on;
plot(t, out.logVol, 'k-', 'LineWidth', 1.5);
grid on; xlabel('Time (s)'); ylabel('log_{10} sqrt(det P)');
title('4-D set volume (log scale)');
legend('predict/update cycle','after update');
subplot(1,3,3);
semilogy(t, out.semiAlong, 'b-', 'LineWidth', 1.4); hold on;
semilogy(t, out.semiCross, 'g-', 'LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('m');
title('Set extent along LOS (range) and across LOS (bearing)');
legend('along LOS','cross LOS');
end

function plotGeometry(out, prm, t)
figure('Name','Fig. 4  Sensing geometry (Grassmannian)','Position',[60 100 1500 700]);
subplot(2,2,1);
plot(t, out.dGr, 'b-', 'LineWidth', 1.2); hold on;
plot(t, sqrt(2)*out.dBeta, 'r--', 'LineWidth', 1);
plot(t, out.nov, 'g-', 'LineWidth', 1);
grid on; xlabel('Time (s)'); ylabel('rad');
title('d_{Gr}(U_{k-1},U_k), sqrt(2)|\Delta\beta|, and bearing novelty [E11]');
legend('d_{Gr}','sqrt(2) |\Delta\beta|','novelty (transported bearing rows)');
subplot(2,2,2);
gbar = nan(size(t));
for k = prm.win:numel(t), gbar(k) = mean(out.dGr(k-prm.win+1:k), 'omitnan'); end
[ax, h1, h2] = plotyy(t, gbar, t, out.Vw);
set(h1, 'LineWidth', 1.3);  set(h2, 'LineWidth', 1.3);
hold(ax(1), 'on');  plot(ax(1), [t(1) t(end)], [prm.gRef prm.gRef], 'k--');
ylabel(ax(1), 'windowed d_{Gr} (rad)');  ylabel(ax(2), sprintf('own-ship speed cmd (m/s), %s', prm.guidance));
xlabel('Time (s)'); grid on;
title(sprintf('Adaptation: windowed d_{Gr} tracks g_{ref} = %.3f', prm.gRef));
subplot(2,2,3);
plot(t, out.rank2, 'r-', 'LineWidth', 1.4); hold on;
plot(t, out.rank3 + 0.05, 'g-', 'LineWidth', 1.4);
grid on; ylim([1.5 4.5]); xlabel('Time (s)'); ylabel('rank');
title('dim(U_{k-1}+U_k) and dim(U_{k-2}+U_{k-1}+U_k)');
legend('2-step window','3-step window (offset)');
subplot(2,2,4);
plot(t, out.rTrue, 'k-', 'LineWidth', 1.5); hold on;
plot(t, out.rEst, 'b--', 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('m');
title('Range to target: true vs estimated centre');
legend('true range','estimated range');
end

function compareN(Ncompare, nTrials, prm, mode)
nC = numel(Ncompare);  col = lines(nC);
fprintf('\n    N   final RMS pos (m)   final RMS vel (m/s)   final area (m^2)   containment (%%)   own-ship max (m)\n');
figure('Name','Fig. 5  Dependence on N','Position',[70 120 1500 780]);
for i = 1:nC
    Ni = Ncompare(i);
    tgt = generateTarget(Ni, prm, mode);
    PE = zeros(nTrials,Ni);  VE = PE;  AR = PE;  QQ = PE;  ownMax = 0;  xOwn1 = [];
    for tr = 1:nTrials
        o = runSMF(Ni, prm, tgt, 100+tr);
        PE(tr,:) = o.posErr;  VE(tr,:) = o.velErr;  AR(tr,:) = o.area;  QQ(tr,:) = o.q;
        ownMax = max(ownMax, max(hypot(o.xOwn(1,:), o.xOwn(3,:))));
        if tr == 1, xOwn1 = o.xOwn; end
    end
    tt = (0:Ni-1)*prm.T;
    rp = sqrt(mean(PE.^2,1));  rv = sqrt(mean(VE.^2,1));  ma = mean(AR,1);
    lw = 2.6 - 0.45*(i-1);
    subplot(2,2,1); semilogy(tt, rp, 'Color', col(i,:), 'LineWidth', lw); hold on;
    subplot(2,2,2); semilogy(tt, rv, 'Color', col(i,:), 'LineWidth', lw); hold on;
    subplot(2,2,3); semilogy(tt, ma, 'Color', col(i,:), 'LineWidth', lw); hold on;
    subplot(2,2,4); plot(xOwn1(1,:), xOwn1(3,:), 'Color', col(i,:), 'LineWidth', lw); hold on;
    tail = max(1,Ni-9):Ni;
    fprintf('%5d   %16.2f   %18.3f   %16.3e   %14.2f   %14.1f\n', Ni, mean(rp(tail)), ...
        mean(rv(tail)), mean(ma(tail)), 100*mean(QQ(:) <= 1+1e-6), ownMax);
end
names = arrayfun(@(n) sprintf('N = %d', n), Ncompare, 'UniformOutput', false);
subplot(2,2,1); grid on; xlabel('Time (s)'); ylabel('m');   title(sprintf('RMS position error (%d trials)', nTrials)); legend(names);
subplot(2,2,2); grid on; xlabel('Time (s)'); ylabel('m/s'); title('RMS velocity error'); legend(names);
subplot(2,2,3); grid on; xlabel('Time (s)'); ylabel('m^2'); title('Mean position-ellipse area'); legend(names);
subplot(2,2,4); th = linspace(0,2*pi,200);
plot(prm.Rmax*cos(th), prm.Rmax*sin(th), 'r--', 'LineWidth', 1.2); plot(0,0,'k^','MarkerFaceColor','k');
axis equal; grid on; xlabel('East (m)'); ylabel('North (m)'); title('Own-ship trajectory');
legend([names, {sprintf('%d m limit', prm.Rmax), 'start'}], 'Location','southoutside');
end

function plotConvergence(out, t)
% Convergence of the set estimate: (a) position ellipses re-centred on the
% truth in the line-of-sight frame, (b) all four semi-axes sqrt(lambda_i(P)),
% (c) volume across every predict/update, with a count of any increases.
% With aT = 0 the volume is non-increasing: prediction has det(Phi) = 1 and
% each cut can choose rho -> 0 (no cut), so the optimal cut never grows it.
N = numel(t);
figure('Name','Fig. 6  Convergence of the ellipsoids','Position',[80 140 1500 480]);
ramp = [0.72 0.83 0.96; 0.53 0.71 0.94; 0.33 0.60 0.91; 0.16 0.47 0.84; 0.11 0.36 0.67; 0.05 0.21 0.42];
ks = unique(max(1, round(N*[0.025 0.05 0.125 0.25 0.5 1])));
subplot(1,3,1); hold on;
th = linspace(0, 2*pi, 120);  hL = zeros(1,numel(ks));  lab = cell(1,numel(ks));
for i = 1:numel(ks)
    k = ks(i);  mu = out.mu(:,k);  x = out.xRel(:,k);
    los = mu([1 3])/norm(mu([1 3]));
    R = [los(2) -los(1); los(1) los(2)];           % rows: cross-LOS, along-LOS
    Pl = R*out.P([1 3],[1 3],k)*R';  c = R*(mu([1 3]) - x([1 3]));
    [V,D] = eig((Pl+Pl')/2);
    pts = V*diag(sqrt(max(diag(D),0)))*[cos(th); sin(th)] + repmat(c,1,numel(th));
    hL(i) = plot(pts(1,:), pts(2,:), '-', 'Color', ramp(min(i,6),:), 'LineWidth', 2);
    lab{i} = sprintf('t = %.0f s', t(k));
end
hT = plot(0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
grid on; xlabel('cross-LOS (m)'); ylabel('along-LOS / range (m)');
title('(a) Position set around the true target');
legend([hL hT], [lab {'true target'}], 'Location', 'northeast');
subplot(1,3,2);
ev = zeros(4,N);
for k = 1:N, ev(:,k) = sqrt(sort(max(eig((out.P(:,:,k)+out.P(:,:,k)')/2),0), 'descend')); end
semilogy(t, ev', 'LineWidth', 1.8);
grid on; xlabel('Time (s)'); ylabel('sqrt(\lambda_i(P))  (mixed m, m/s)');
title('(b) Semi-axes of the 4-D ellipsoid');
legend('axis 1','axis 2','axis 3','axis 4');
subplot(1,3,3);
tt = reshape([t; t], 1, []);
vv = reshape([out.logVolPrior; out.logVol], 1, []);
nInc = sum(diff(vv) > 1e-6);
plot(tt, vv, 'Color', [0.6 0.6 0.6]); hold on;
plot(t, out.logVol, 'k-', 'LineWidth', 1.8);
grid on; xlabel('Time (s)'); ylabel('log_{10} sqrt(det P)');
title(sprintf('(c) Set volume (%d increases > 1e-6)', nInc));
legend('every predict/update','after update');
end