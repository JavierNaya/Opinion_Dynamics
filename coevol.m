k = 4;
gamma = 10;
phi = 1;

N = 100;   % number of people
M = round(N*k/2);
G = round(N/gamma);

x = randi(G,N,1);   % range of opinions
m = zeros(N);   % symmetric matrix specifying edges between nodes

l = M;    %   initiate randomly the edges
while l > 0
    i = randi(N);
    j = randi(N);
    if m(i,j) == 0 && i ~= j
        m(i,j) = 1;
        m(j,i) = 1;
        l = l-1;
    end
end

evo = x;   % evolution data
t_end = 1e3;

 for t = 1:t_end
     i = randi(N);
      if sum(m(i,:)) ~= 0    % do nothing if i has no edges
        h = []; % same opinions
        for f = 1:N
            if x(f) == x(i) && f ~= i
                h = [h f];
            end
        end
        c = []; % other end of vertices
        for r = 1:N
            if m(i,r) == 1
                c = [c r];
            end
        end
        if rand <= phi && isempty(h) == 0
            q = randi(length(c));
            j = randi(length(h));
            if m(i,h(j)) == 0
                m(i,c(q)) = 0;
                m(c(q),i) = 0;
                m(i,h(j)) = 1;
                m(h(j),i) = 1;
            else
                t = t-1;
            end
        else
            j = randi(length(c));
            x(i) = x(c(j));
        end
      end
      evo = [evo x];
 end
 
 figure
 hist(evo(:,t_end))
 
%  figure
%  hold on
%  for s = 1:N
%      plot(0:t_end,evo(s,:))
%  end
