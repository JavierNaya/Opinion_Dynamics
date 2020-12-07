function m = fj(n,x,t_end,gg)

m = x;

A = rand(n);
for i = 1:n
    A(i,:) = A(i,:) / sum(A(i,:));
end
G = diag(rand(n,1)*gg);
I = eye(n);

for t = 2:t_end+1
    m(:,t) = G*m(:,1) + (I-G)*A*m(:,t-1);
end
