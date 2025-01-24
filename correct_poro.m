function [data] = correct_poro(filepath,stomatal_sidedness)
%ADD_GSW_CORRECTION_TO_LI600 Applies the Bailey & Rizzo (2024) correction
%of chamber air temperature and stomatal conductance to a csv file exported
%from an LI-600

% Input:
%  - filepath: Path to the CSV file exported from LI-600 (required).
%  - stomatal_sidedness: Correction factor for stomatal sidedness ...
%    1 if hypostomatous, 2 if amphistomatous, or anywhere in between (optional, default = 1).

% Output:
%  - new csv file with corrected gsw, T_chamber, W_chamber
if nargin < 2 || isempty(stomatal_sidedness)
    stomatal_sidedness = 1;  % Default
end

data = readtable(filepath);
try
    size(data.gsw);
catch ME
    opts = detectImportOptions(filepath);
    opts.VariableNamesLine = 2;
    opts.DataLines = [4 inf];
    data = readtable(filepath, opts);
end

sidedness = stomatal_sidedness*ones(size(data.gsw));
T_chambs = zeros(size(data.gsw));
T_outs = zeros(size(data.gsw));
W_chambs = zeros(size(data.gsw));
gsw_total = data.gsw.*sidedness;

for i=1:length(data.gsw)
    % -- input --%
    T_in = data.Tref(i);                        % C (inlet air temp)
    T_leaf = data.Tleaf(i);                     % C (leaf temp)

    RH_in = data.rh_r(i)/100;                   % Decimal (inlet RH)
    RH_out = data.rh_s(i)/100;                  % Decimal (outlet RH)     

    u_in = data.flow(i)*1e-6;                   % mol/s (inlet air flow)
    high_flow = 150*1e-6;                       % mol/s (highest inlet air flow setting)
    P_atm = data.P_atm(i);                      % kPa (air pressure)
    s = 0.441786*0.01^2;                        % m^2 (leaf area)
    gbw = 2.921;                                % mol/m^2/s (boundary layer conductance)
    C = 0.03*(u_in./high_flow).^(2.718/2);      % J/s/C (empirical thermal conductance)
    %C = 0.03;                                  % J/s/C (empirical thermal conductance)


    cpa = 29.14;                                % J/mol/C (air heat capacity)
    cpw = 33.5;                                 % J/mol/C (water heat capacity)
    lambdaw = 45502;                            % J/mol (water latent heat of vaporization)

    a = 0.61365;                                % unitless (empirical magnitude of es vs T)
    b = 17.502;                                 % unitless (empirical slope of es vs T)
    c = 240.97;                                 % C (empirical offset of es vs T)

    es = @(T) a*exp(b*T./(T+c));                % kPa (saturation vapor pressure vs T function)
    W = @(T,RH) es(T).*RH./(P_atm);              % mol/mol (water vapor mole fraction)
    Wd = @(T,RH) es(T).*RH./(P_atm-es(T).*RH);   % mol/mol (humidity ratio)
    h = @(T,RH) cpa*T+Wd(T,RH).*(lambdaw + cpw*T); % J/mol (enthalpy) 

    % -- computation -- %
    syms Tout E gsw;
    % -- ASSUMPTION:The chamber air temperature is the average of the inlet and outlet air temperatures -- %
    T_chamb = 0.5*(T_in+Tout);
    % -- ASSUMPTION:The chamber relative humidity is the average of the inlet and outlet relative humidites -- %
    RH_chamb = 0.5*(RH_in+RH_out);
    W_chamb = W(T_chamb,RH_chamb);
    W_in = W(T_in,RH_in);
    W_out = W(Tout,RH_out);
    W_leaf = W(T_leaf,1.0);
    h_in = h(T_in,RH_in);
    h_out = h(Tout,RH_out);
    Q = C.*(T_in - T_chamb);
    gtw = (gsw*gbw)./(gsw+gbw);
    % -- solve implicit system of equations (4,7,8) from Bailey and Rizzo (2024) for T_out, E, gsw -- %
    eq1 = ( E == gtw.*(W_leaf-W_chamb) );
    eq2 = ( E == s.^(-1)*u_in*(W_out - W_in)*(1-W_out).^(-1) );
    eq3 = ( E == s.^(-1).*((Q + u_in.*h_in)./(h_out)-u_in) );
    [T_out, E_out, gsw_out]  = vpasolve(eq1,eq2,eq3,Tout,E,gsw);
   
    E = double(E_out);                            % mol/m^2/s
    gsw_bottom(i) = double(gsw_out);              % mol/m^2/s
    gsw_total(i) = gsw_bottom(i).*sidedness(i);   % mol/m^2/s
    T_chambs(i) = subs(T_chamb,T_out);            % C
    T_outs(i) = T_out;                            % C
    W_chambs(i) = double(subs(W_chamb,T_out));    % mol/mol
end

data.gsw_corrected = gsw_total;
data.Ta_chamb_corrected = T_chambs;
data.T_out_corrected = T_outs;
data.W_chamb_corrected = W_chambs;
data.stomatal_sidedness = sidedness;

[p,f,e]=fileparts(filepath);
filename=fullfile(p,f);
writetable(data,filename+"_corrected"+e);

end
