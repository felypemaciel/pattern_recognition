function data = boxcox_transf(data)
    [r, c] = size(data);

    initial_lambda = 0;

    optimal_lambda = zeros(r,1);

    for i = 1:r
        V = data(i,:)';
        log_likelihood = @(lambda) -sum(log(boxcox(V, lambda)));
        optimal_lambda(i) = fminsearch(log_likelihood, initial_lambda);
    end

    for i = 1:r
        if optimal_lambda(i) ~= 0
            data(i,:) = (data(i,:).^optimal_lambda(i) - 1)/optimal_lambda(i);

        else
            data(i,:) = log(data(i,:));

        end
    end
end