p = 50;      % nb of people
op = 2;     % range of opinions

x0 = (rand(p,1)-0.5)*op;    % continuous random op. distribution
if size(x0,1) == 1
    x0 = x0';
end

t_end = 10;       % time of simulation
x = [x0, zeros(p,t_end-1)];

%x = cm(p,x,t_end);
%x = fj(p,x,t_end,0.5);
%x = tv(p,x,t_end);
x = bc(p,x,t_end,0.35,0.1);

figure
hold on
for j = 1:p
    plot(0:t_end,x(j,:))
    %set(gca,'XScale','log')
end
xlabel('Time')
ylabel('Opinion range')