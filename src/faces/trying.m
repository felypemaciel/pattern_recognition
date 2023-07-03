data = load("recfaces.dat");

size(data)

X = data(:,1:end-1)';
Y = data(:,end);