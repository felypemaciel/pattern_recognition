close all

data = load("recfaces.dat");

size(data)

X = data(:,1:end-1)';
Y = data(:,end);

x = X(:,4);
lamb = 0.05;
xstar = (x.^lamb - 1)/lamb;
figure; 

mxstar = mean(xstar);
dpxstar = std(xstar);
xnorm = normrnd(mxstar, dpxstar, 5000, 1);

subplot(2,1,1);
histfit(xstar);
subplot(2,1,2);
cdfplot(xstar);
hold on;
cdfplot(xnorm);
title("\lambda = ", num2str(lamb))