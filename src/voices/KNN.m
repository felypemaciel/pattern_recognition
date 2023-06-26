function [STATS, hit_perc, X, M] = KNN(data, Nr, Ptrain)

X = data(:,1:end-1);
Y = data(:,end);

med = mean(X,2);   % Media dos atributos
dp = std(X,[],2);  % desvio padrao dos atributos
X = (X-med)./dp; 

[N_data, ~] = size(X);      % number of examples
C = max(Y);                 % number of classes

N_train = floor((Ptrain/100)*N_data);

for r = 1:Nr

    I = randperm(N_data);   % shuffles indexes of columns of x
    X = X(I,:);             % shuffles the columns of x
    Y = Y(I,:);             % takes the labels to the same positions

    X_train = X(1:N_train,:);   % train data
    Y_train = Y(1:N_train,:);   % train labels

%     for k = 1:C
%         I = find(Y_train == k);         % finds columns whose vectors are 
%                                         % from class k
%         M{k} = mean(X_train(I,:),2);    % calculates the centroid of the 
%                                         % subsets of vectors from class k
%         label{k} = k;                   % labels the 
%     end

    X_test = X(N_train+1:end,:);    % test data
    Y_test = Y(N_train+1:end,:);    % test labels

    N_test = N_data - N_train;       % number of test data

    hit = 0;    % hist counter

    for i = 1:N_test
        X_new = X_test(i,:);    % test vector
        Y_new = Y_test(i);      % correct test label[

        for j = 1:N_train
            distances(j) = norm(X_new - X_train(j,:));
        end

        [D_min, J_min] = min(distances);

        if Y_train(J_min) == Y_new
            hit = hit+1;
        end
    end

    hit_perc(r) = 100*hit/N_test;
end

STATS = [mean(hit_perc), min(hit_perc), max(hit_perc), median(hit_perc), std(hit_perc)];