function m = bc(n,x,t_end,eps,stoc)

m = x;

for t = 2:t_end+1
    A = ones(n);
    for i = 1:n
        for j = 1:n
               if abs(m(i,t-1)-m(j,t-1)) > eps
                   A(i,j) = 0;
               end
%              A(i,j) = abs(m(i,t-1)-m(j,t-1));
%               if m(i,t-1)-m(j,t-1) ~= 0
%                   A(i,j) = 1/abs(m(i,t-1)-m(j,t-1));
%               end
        end
    end
    for j = 1:n
        A(j,:) = A(j,:) / sum(A(j,:));
    end
    m(:,t) = A * m(:,t-1);
    m(:,t) = m(:,t) + (rand(n,1)-0.5)*stoc;
end
