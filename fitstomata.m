%clear;
data = readtable("poro/Romaine_poro.csv");
data = correct_poro("poro/Romaine_poro.csv");
figure();
x = data.Qamb*0.85;
y = data.VPDleaf*1000./data.P_atm;
z = data.gsw_corrected;
[res,gof] = fit([x,y],z,"Em*x./(k+b*x+(x+i0).*y)","Lower",[0 0 0 0]);
plot(res); hold on;
scatter3(x,y,z,"filled","o","k");
xlabel("Q","Interpreter","latex");
ylabel("D","Interpreter","latex");
zlabel("g$_{sw}$","Interpreter","latex");
title("Romaine var. Bondi SS","Interpreter","latex")
set(gca,"FontSize",15)
res
%%
