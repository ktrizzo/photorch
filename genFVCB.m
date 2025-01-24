clear;
spps = ["Romaine","Iceberg"];
vars = ["var. Bondi SS","var. Calmar"];

for i=1:length(spps)
    species = spps(i);
    var = vars(i);
    P = readtable(species+"Params.csv");

    figure("Position",[100 100 700 300]);

    % A vs T @ Q2000, Ci 0.7*420
    Ci = linspace(0,2000,60);
    T = (35+273.15)*ones(size(Ci));
    Q = 2000*ones(size(T));
    [Ci,Q,T] = meshgrid(Ci,Q,T);
    subplot(1,3,1);
    for i = 1:length(unique(P.species))
        p = P(i,:);
        x = [Ci(:),Q(:),T(:)];
        A = fvcb(x,p);
        A = reshape(A,[60,60,60]);
        x = Ci(1,1:60,1);
        y = A(1,1:60,1);
        plot(x(:),y(:),linewidth=6); hold on;
    end
    xlabel("Ci","Interpreter","latex");
    ylabel("A","Interpreter","latex");
    set(gca,"FontSize",13);
    set(gca,"LineWidth",2);
    ylim([0 75]);


    % A vs Q @ T25, Ci 0.7*420
    Q = linspace(0,2000,60);
    T = (25+273.15)*ones(size(Q));
    Ci = 2000*ones(size(T));

    [Ci,Q,T] = meshgrid(Ci,Q,T);
    subplot(1,3,2);
    for i = 1:length(unique(P.species))
        p = P(i,:);
        x = [Ci(:),Q(:),T(:)];
        A = fvcb(x,p);
        A = reshape(A,[60,60,60]);
        x = Q(1:60,1,1);
        y = A(1:60,1,1);
        plot(x(:),y(:),linewidth=6); hold on;
    end
    xlabel("Q","Interpreter","latex");
    ylabel("A","Interpreter","latex");
    set(gca,"FontSize",13);
    set(gca,"LineWidth",2);
    title(species+" "+var,"FontSize",15,"Interpreter","latex");
    ylim([0 75]);

    % A vs T @ Q2000, Ci 0.7*420
    T = linspace(10,45,60)+273.15;
    Ci = 2000*ones(size(T));
    Q = 2000*ones(size(T));
    [Ci,Q,T] = meshgrid(Ci,Q,T);
    subplot(1,3,3);
    for i = 1:length(unique(P.species))
        p = P(i,:);
        x = [Ci(:),Q(:),T(:)];
        A = fvcb(x,p);
        A = reshape(A,[60,60,60]);
        x = T(1,1,1:60)-273.15;
        y = A(1,1,1:60);
        plot(x(:),y(:),linewidth=6); hold on;
    end

    xlabel("T","Interpreter","latex");
    ylabel("A","Interpreter","latex");
    ylim([0 75]);
    set(gca,"FontSize",13);
    set(gca,"LineWidth",2);
    set(gcf,"Color","white");
end


%%
for i=1:length(spps)
    species = spps(i);
    var = vars(i);
data = readtable(species+"Curves.csv");

P = readtable(species+"Params.csv");
Ci = linspace(100,2000,60);
T = linspace(273,40+273,60);
[Ci,T] = meshgrid(Ci,T);
Q = 2000*ones(size(T));
figure();
subplot(1,2,1);
p = P(1,:);
x = [Ci(:),Q(:),T(:)];
A = fvcb(x,p);
A = reshape(A,[60,60]);
x=Ci;
y=T;
z=real(A);
surf(x,y-273.15,z); hold on;
xlabel("Ci","Interpreter","latex");
ylabel("T","Interpreter","latex");
zlabel("A","Interpreter","latex");
set(gca,"FontSize",13);
set(gca,"LineWidth",2);
hold on;
scatter3(data.Ci,data.Tleaf,data.A,"filled","r");
shading interp;
view([95 5]);


% gif_filename = 'rotating_plot.gif';
% n_frames = 60;        % Number of frames for the GIF
% view_angle = linspace(0, 360, n_frames);  % Angles for rotation
% 
% % Loop to rotate the plot and capture frames
% for i = 1:n_frames
%     view(view_angle(i), 5);  % Rotate the plot (azimuth, elevation)
% 
%     % Capture the current figure as a frame
%     frame = getframe(gcf);
%     img = frame2im(frame);
%     [img_ind, cmap] = rgb2ind(img, 256);
% 
%     % Write to GIF
%     if i == 1
%         imwrite(img_ind, cmap, gif_filename, 'gif', 'LoopCount', inf, 'DelayTime', 0.1);
%     else
%         imwrite(img_ind, cmap, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%     end
% end



subplot(1,2,2);
data = readtable(species+"Curves.csv");

P = readtable(species+"Params.csv");
Ci = linspace(5,2000,60);
Q = linspace(0,2000,60);
[Ci,Q] = meshgrid(Ci,Q);
T = 298.15*ones(size(Ci));




p = P(1,:);
x = [Ci(:),Q(:),T(:)];
A = fvcb(x,p);
A = reshape(A,[60,60]);
x=Ci;
y=Q;
z=real(A);
surf(x,y,z); hold on;
xlabel("Ci","Interpreter","latex");
ylabel("Q","Interpreter","latex");
zlabel("A","Interpreter","latex");
set(gca,"FontSize",13);
set(gca,"LineWidth",2);
hold on;
scatter3(data.Ci,data.Qin,data.A,"filled","r");
shading interp;
view([95 5]);

sgtitle(species+" "+var);
end
%%
colors = [[158,1,66];[213,62,79];[244,109,67];[253,174,97];[254,224,139]; ...
    [230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
natives = [repmat([0 0 0],5,1);repmat([0.5 0.5 0.5],5,1)];
figure();

% Define variables for response curves
Q = linspace(0, 2000, 60);    % Fixed light intensity
T = 313 * ones(size(Q));     % Fixed temperature
Ci = 0.7*420*ones(size(Q)); % Intercellular CO2 range

% Load species data
P = readtable("data/fvcbparams.csv");
species_list = ["almond" "grape" "olive" "pistachio" "walnut"];
figure();
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(Ci));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(Ci)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(Q, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 20]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)


figure()
species_list = ["big leaf maple","blue elderberry","bay","toyon","western redbud"];
colors = [[230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(Ci));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(Ci)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(Q, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 20]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)

%%
%Ci
%%
colors = [[158,1,66];[213,62,79];[244,109,67];[253,174,97];[254,224,139]; ...
    [230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
natives = [repmat([0 0 0],5,1);repmat([0.5 0.5 0.5],5,1)];
figure();

% Define variables for response curves
Ci = linspace(0, 2000, 60);    % Fixed light intensity
T = 313 * ones(size(Ci));     % Fixed temperature
Q = 2000*ones(size(Ci)); % Intercellular CO2 range

% Load species data
P = readtable("data/fvcbparams.csv");
species_list = ["almond" "grape" "olive" "pistachio" "walnut"];
figure();
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(Ci));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(Ci)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(Ci, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 55]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)


figure()
species_list = ["big leaf maple","blue elderberry","bay","toyon","western redbud"];
colors = [[230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(Ci));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(Ci)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(Ci, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 55]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)

%%
%%
colors = [[158,1,66];[213,62,79];[244,109,67];[253,174,97];[254,224,139]; ...
    [230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
natives = [repmat([0 0 0],5,1);repmat([0.5 0.5 0.5],5,1)];
figure();

% Define variables for response curves
T = linspace(0, 60, 60)+273.15;     % Fixed temperature
Ci = 0.7*420*ones(size(T));    % Fixed light intensity
Q = 2000*ones(size(Ci)); % Intercellular CO2 range

% Load species data
P = readtable("data/fvcbparams.csv");
species_list = ["almond" "grape" "olive" "pistachio" "walnut"];
figure();
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(T));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(T)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(T-273.15, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 30]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)


figure()
species_list = ["big leaf maple","blue elderberry","bay","toyon","western redbud"];
colors = [[230,245,152];[171,221,164];[102,194,165];[50,136,189];[94,79,162]]./255;
% Iterate through each species
for i = 1:length(species_list)
    % Extract species-specific parameters
    species_name = species_list{i};
    p = P(strcmp(P.species, species_name), :);

    % Pre-allocate assimilation array for this species
    A = zeros(size(Ci));
    
    % Compute assimilation for each value of Ci
    for j = 1:length(Ci)
        % Prepare input data for fvcb function
        x = [Ci(j), Q(j), T(j)];
        A(j) = fvcb(x, p);
    end

    % Plot the response curve for the current species
    plot(T-273.15, A, 'LineWidth', 8, 'DisplayName', species_name,'Color',colors(i,:));
    hold on;
end

% Customize the plot
%xlabel('Light Flux Q ($\mu$mol m$^{-2}$ s$^{-1}$)');
%ylabel('Assimilation Rate A ($\mu$mol m$^{-2}$ s$^{-1}$)');
ylim([-5 30]);
set(gca,"FontSize",30)
%title('Assimilation Light Response');
legend('hide', 'Location', 'Best');
set(gca,"LineWidth",5)

%%
clear;
data = readtable("curves/2025-01-09_Romaine_Bondi/2025-01-09-1037_aci_T25_Q2000.txt","NumHeaderLines",66);
data = data(2:end,:);