% clear; clc; close all;

D=load('recvoz.dat');

Nr = 100;  % No. de repeticoes

Ptrain = 80; % Porcentagem de treinamento

tic; [st_quadratic TX_OK0 X0 m0 S0 posto0] = quadratico(D,Nr,Ptrain); tquad=toc;    % One COV matrix per class
tic; [st_quadratic_v1 TX_OK1 X1 m1 S1 posto1] = variante1(D,Nr,Ptrain,0.01); tquadv1=toc; % Regularization method 1 (Tikhonov)
tic; [st_quadratic_v2 TX_OK2 X2 m2 S2 posto2] = variante2(D,Nr,Ptrain); tquadv2=toc;     % One common COV matrix (pooled)
tic; [st_quadratic_v3 TX_OK3 X3 m3 S3 posto3] = variante3(D,Nr,Ptrain,0.5); tquadv3=toc; % Regularization method 2 (Friedman)
tic; [st_quadratic_v4 TX_OK4 X4 m4 S4 posto4] = variante4(D,Nr,Ptrain); tquadv4=toc;     % Naive Bayes Local (Based on quadratico)
tic; [st_linear_mq TX_OK5 W] = linearMQ(D,Nr,Ptrain,'none'); tlinearmq=toc;     % Classificador Linear de Minimos Quadrados
tic; [st_dmc TX_OK6 X6] = DMC_fm(D, Nr, Ptrain); tdmc = toc;
tic; [st_knn TX_OK7 X7] = KNN(D, Nr, Ptrain); tknn = toc;
tic; [st_max_corr TX_OK7 X7] = max_corr(D, Nr, Ptrain); tmaxcorr = toc;

st_quadratic
st_quadratic_v1
st_quadratic_v2
st_quadratic_v3
st_quadratic_v4
st_linear_mq
st_max_corr
st_dmc
st_knn

TEMPOS=[tquad tquadv1 tquadv2 tquadv3 tquadv4 tlinearmq tmaxcorr tdmc tknn]'

boxplot([TX_OK0' TX_OK1' TX_OK2' TX_OK3' TX_OK4' TX_OK5'])
set(gca (), "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"Quadratico","Variante 1", "Variante 2","Variante 3","Variante 4","MQ"})
title('Conjunto Escolhido');
xlabel('Classificador');
ylabel('Taxas de acerto');