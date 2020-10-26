function m = tv(n,x,t_end)

m = x;

for t = 2:t_end+1
    A = rand(n);
    for i = 1:n
        A(i,:) = A(i,:) / sum(A(i,:));
    end
    m(:,t) = A * m(:,t-1);
end