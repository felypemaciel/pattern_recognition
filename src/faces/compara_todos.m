clear; clc;

D=load('recfaces.dat');

Nr=50;  % No. de repeticoes

Ptrain=80; % Porcentagem de treinamento

tic; [STATS_0, TX_OK0, X0, m0, S0, posto0]=quadratico(D,Nr,Ptrain); Tempo0=toc;    % One COV matrix per class
tic; [STATS_1, TX_OK1, X1, m1, S1, posto1]=variante1(D,Nr,Ptrain,0.01); Tempo1=toc; % Regularization method 1 (Tikhonov)
tic; [STATS_2, TX_OK2, X2, m2, S2, posto2]=variante2(D,Nr,Ptrain); Tempo2=toc;     % One common COV matrix (pooled)
tic; [STATS_3, TX_OK3, X3, m3, S3, posto3]=variante3(D,Nr,Ptrain,0.5); Tempo3=toc; % Regularization method 2 (Friedman)
tic; [STATS_4, TX_OK4, X4, m4, S4, posto4]=variante4(D,Nr,Ptrain); Tempo4=toc;     % Naive Bayes Local (Based on quadratico)
tic; [STATS_5, TX_OK5, W]=linearMQ(D,Nr,Ptrain,'none'); Tempo5=toc;     % Classificador Linear de Minimos Quadrados
tic; [STATS_MC, TX_OK_MC, W_MC] = max_corr(D,Nr, Ptrain); Tempo_MC = toc;
tic; [STATS_DMC, TX_OK_DMC, W_DMC] = DMC_fm(D,Nr, Ptrain); Tempo_DMC = toc;
tic; [STATS_KNN, TX_OK_KNN, W_KNN] = KNN(D,Nr, Ptrain); Tempo_KNN = toc;

STATS_0 = [STATS_0, Tempo0]
STATS_1 = [STATS_1, Tempo1]
STATS_2 = [STATS_2, Tempo2]
STATS_3 = [STATS_3, Tempo3]
STATS_4 = [STATS_4, Tempo4]
STATS_5 = [STATS_5, Tempo5]
STATS_MC = [STATS_MC, Tempo_MC]
STATS_DMC = [STATS_DMC, Tempo_DMC]
STATS_KNN = [STATS_KNN, Tempo_KNN]


TEMPOS = [Tempo0 Tempo1 Tempo2 Tempo3 Tempo4 Tempo5 Tempo_MC Tempo_DMC Tempo_KNN]';

figure;
boxplot([TX_OK0' TX_OK1' TX_OK2' TX_OK3' TX_OK4' TX_OK5'])
set(gca (), "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"Quadratico","Variante 1", "Variante 2","Variante 3","Variante 4","MQ"})
title('Conjunto Coluna');
xlabel('Classificador');
ylabel('Taxas de acerto');


