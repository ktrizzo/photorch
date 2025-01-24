function [a] = fvcb(x,p)
% Constants
R = 0.008314;

% Define Tresp function
Tresp = @(T, dHa, dHd, Topt) exp(dHa/R * (1/298 - 1./T)) .* ...
        (1 + exp(dHd/R * (1/Topt - 1/298) - log(dHd/dHa - 1))) ./ ...
        (1 + exp(dHd/R * (1/Topt - 1./T) - log(dHd/dHa - 1)));

% Functions dependent on Tresp
Vcmax = @(T) p.Vcmax25 * Tresp(T, p.Vcmax_dHa, p.Vcmax_dHd, p.Vcmax_Topt);
Jmax = @(T) p.Jmax25 * Tresp(T, p.Jmax_dHa, p.Jmax_dHd, p.Jmax_Topt);
Kc = @(T) p.Kc25 * Tresp(T, p.Kc_dHa, 500, 1000);
Ko = @(T) p.Ko25 * Tresp(T, p.Ko_dHa, 500, 1000);
Gamma = @(T) p.Gamma25 * Tresp(T, p.Gamma_dHa, 500, 1000);
Rd = @(T) p.Rd25 * Tresp(T, p.Rd_dHa, 500, 1000);
Kco = @(T) Kc(T) .* (1 + p.O ./ Ko(T));

% Light response function J
a = max(p.theta, 0.0001); % Ensure 'a' is not zero
ia = 1 ./ a; % Reciprocal of a
J = @(Q, T) (-(-(p.alpha * Q + Jmax(T))) - sqrt((-(p.alpha * Q + Jmax(T))).^2 - 4 * a .* (p.alpha * Q .* Jmax(T)))) * 0.5 .* ia;

% RuBisCO-limited photosynthesis
vr = @(Ci, T) Vcmax(T) .* ((Ci - Gamma(T)) ./ (Ci + Kco(T))) - Rd(T);

% Electron transport-limited photosynthesis
jr = @(Ci, Q, T) 0.25 * J(Q, T) .* ((Ci - Gamma(T)) ./ (Ci + 2 * Gamma(T))) - Rd(T);

% Minimum of vr and jr
hmin = @(f1, f2) (f1 + f2 - sqrt((f1 + f2).^2 - 4 * 0.999 * f1 .* f2)) / (2 * 0.999);

% Net assimilation rate
A = @(Ci, Q, T) hmin(vr(Ci, T), jr(Ci, Q, T));

% Inputs
Ci = x(:, 1);
Q = x(:, 2);
T = x(:, 3);

% Compute assimilation rates
a = A(Ci, Q, T);
end

