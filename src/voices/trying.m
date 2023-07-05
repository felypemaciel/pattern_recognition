close all

data = load("recvoz.dat");

X = data(:,1:end-1)';
Y = data(:,end);

x = X(:,4);
lamb = 0.06;
xstar = (x.^lamb - 1)/lamb; 

mxstar = mean(xstar);
dpxstar = std(xstar);
xnorm = normrnd(mxstar, dpxstar, 5000, 1);

% subplot(1,2,1);
figure;
histfit(xstar);
title('Histograma de um dos atributos do conjunto')
% subplot(1,2,2);

figure;
cdfplot(xstar);
hold on;
cdfplot(xnorm)
title("Comparação das CDF para \lambda = 0.06")