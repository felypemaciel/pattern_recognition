function newData = boxcox_maciel(data)
    data = data';

    [r, c] = size(data);

    newData = zeros(r,c);
    lambdas = zeros(1,c);

    for i=1:c
        [newData(:,i), lambdas(i)] = boxcox(data(:,i));
    end
end