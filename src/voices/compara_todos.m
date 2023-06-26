% clear; clc; close all;

D=load('recvoz.dat');

Nr = 100;  % No. de repeticoes

Ptrain = 80; % Porcentagem de treinamento

tic; [st_quadratic TX_OK0 X0 m0 S0 posto0] = quadratico(D,Nr,Ptrain); Tempo0=toc;    % One COV matrix per class
tic; [st_quadratic_v1 TX_OK1 X1 m1 S1 posto1] = variante1(D,Nr,Ptrain,0.01); Tempo1=toc; % Regularization method 1 (Tikhonov)
tic; [st_quadratic_v2 TX_OK2 X2 m2 S2 posto2] = variante2(D,Nr,Ptrain); Tempo2=toc;     % One common COV matrix (pooled)
tic; [st_quadratic_v3 TX_OK3 X3 m3 S3 posto3] = variante3(D,Nr,Ptrain,0.5); Tempo3=toc; % Regularization method 2 (Friedman)
tic; [st_quadratic_v4 TX_OK4 X4 m4 S4 posto4] = variante4(D,Nr,Ptrain); Tempo4=toc;     % Naive Bayes Local (Based on quadratico)
tic; [st_quadratic_v5 TX_OK5 W] = linearMQ(D,Nr,Ptrain,'none'); Tempo5=toc;     % Classificador Linear de Minimos Quadrados
tic; [st_dmc TX_OK6 X6] = DMC_fm(D, Nr, Ptrain); toc;
tic; [st_knn TX_OK7 X7] = KNN(D, Nr, Ptrain); toc;
tic; [st_max_corr TX_OK7 X7] = max_corr(D, Nr, Ptrain); toc;

st_quadratic
st_quadratic_v1
st_quadratic_v2
st_quadratic_v3
st_quadratic_v4
st_quadratic_v5
st_dmc
st_knn
st_max_corr

TEMPOS=[Tempo0 Tempo1 Tempo2 Tempo3 Tempo4 Tempo5]

boxplot([TX_OK0' TX_OK1' TX_OK2' TX_OK3' TX_OK4' TX_OK5'])
set(gca (), "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"Quadratico","Variante 1", "Variante 2","Variante 3","Variante 4","MQ"})
title('Conjunto Escolhido');
xlabel('Classificador');
ylabel('Taxas de acerto');