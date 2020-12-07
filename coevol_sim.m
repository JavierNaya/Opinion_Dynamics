%load('2parties.mat')
figure
gif('ue.gif','DelayTime',0.1,'LoopCount',0,'frame',gcf)
for t = 1:50:length(evo)
    histogram(evo(:,t))
    xlim([0 11])
    ylim([0 50])
    xlabel('Opinions')
    ylabel('Occurences')
    title(['t = ', num2str(t-1)])
    gif
    pause(0.001)
end