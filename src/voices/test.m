close all;

data = load("recvoz.dat");
X = data(:,1:end-1);
x = X(1,:);

% lambdas = 0:0.1:1;
% 
% sum_log_like = zeros(length(lambdas));
% 
% for i=1:length(lambdas)
%     sum_log_like(i) = sum(log(boxcox(x,lambdas(i))));
% end

% [newx, lambda] = boxcox(x');
% 
% subplot(1,2,1); histfit(x);
% subplot(1,2,2); histfit(newx);

[r, c] = size(X);

newX = zeros(r,c);
lambdas = zeros(1,i);

for i=1:c
    [x, lambdas(i)] = boxcox(X(:,i));
end