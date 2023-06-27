data = load("recvoz.dat");

X = data(:,1:end-1);

[m, n] = size(X);

lambda = -2:0.1:2;

for i = 1:n
    for j = 1:length(lambda)
        Xstar = (X(:,n).^lambda(j) - 1)/lambda(j);
        maxlikelihood = mle(Xstar);
        lkh(j) = maxlikelihood(1);
    end
    [max_lkh, max_idx] = max(lkh(:,1));
    opt_lambda(n) = lambda(max_idx);
end