function m = cm(n,x,t_end)

m = x;

A = rand(n);
for i = 1:n
    A(i,:) = A(i,:) / sum(A(i,:));
end

for t = 2:t_end+1
    m(:,t) = A * m(:,t-1);
end