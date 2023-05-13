clear; close all; clc;

% data loading
X = load('atributosIRIS.dat');              % attributes
Y = load("rotulosIRIS.dat");                % labels

n_elements = length(Y);                     % number of elements

train_perc = 0.8;                           % percentage of train dataset
n_train = floor(train_perc*n_elements);     % number of train elements

n_test = n_elements - n_train;              % number of test elements

n_rep = 50;                                 % number of repetitions
n_hits = 0;                                 % number of hits
hit_rate = zeros(n_rep);                    % hit rate

for r = 1:n_rep

    random_idx = randperm(n_elements);          % random indexes

    % dataset with random indexes
    X = X(:,random_idx);
    Y = Y(:,random_idx);

    % train dataset
    X_train = X(:,1:n_train);
    Y_train = Y(:,1:n_train);

    % test dataset
    X_test = X(:,n_train+1:end);
    Y_test = Y(:,n_train+1:end);

    distance = ones(n_test,1);                % distance vector

    for i = 1:n_test

        % element to be tested
        x = X_test(:,i);
        y = Y_test(i);

        for j = 1:n_test

            distance(j) = norm(x - X_train(:,j));       % distance

        end

        [min_D, min_idx] = min(distance);               % minimal distance

        if Y_train(min_idx) == y
            n_hits = n_hits + 1;    
        end
    end

    hit_rate(r) = n_hits/n_test;

end

stats = [mean(hit_rate), std(hit_rate), min(hit_rate), max(hit_rate)]