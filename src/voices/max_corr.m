function [STATS, hit_perc, X, M] = DMC_fm(data, Nr, Ptrain)

X = data(:,1:end-1);
Y = data(:,end);

med = mean(X,2);   % Media dos atributos
dp = std(X,[],2);  % desvio padrao dos atributos
X = (X-med)./dp; 

N_data = length(Y);      % number of examples
C = max(Y);                 % number of classes

N_train = floor((Ptrain/100)*N_data);

for r = 1:Nr

    I = randperm(N_data);   % shuffles indexes of columns of x
    X = X(I,:);             % shuffles the columns of x
    Y = Y(I,:);             % takes the labels to the same positions

    X_train = X(1:N_train,:);   % train data
    Y_train = Y(1:N_train,:);   % train labels

    for k = 1:C
        I = find(Y_train == k);         % finds columns whose vectors are 
                                        % from class k
        [idx, M{k}] = kmeans(X_train(I,:),1);  % calculates the centroid of the 
                                        % subsets of vectors from class k
        label{k} = k;                   % labels the 
    end

    X_test = X(N_train+1:end,:);    % test data
    Y_test = Y(N_train+1:end,:);    % test labels

    N_test = N_data - N_train;       % number of test data

    hit = 0;    % hist counter

    for i = 1:N_test
        X_new = X_test(i,:);    % test vector
        Y_new = Y_test(i);      % correct test label

        for k = 1:C
            norm_M = norm(M{k});
            tilde_M_k = M{k}/norm_M
            tilde_x_new = X_new/norm(X_new)
            correlation_XM(k) = tilde_M_k'*tilde_x_new;
        end

        [D_min, K_min] = max(correlation_XM);

        if K_min == Y_new
            hit = hit+1;
        end
    end

    hit_perc(r) = 100*hit/N_test;
end

STATS = [mean(hit_perc), min(hit_perc), max(hit_perc), median(hit_perc), std(hit_perc)];