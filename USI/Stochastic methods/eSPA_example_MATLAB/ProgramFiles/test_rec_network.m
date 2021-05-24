ind_K=5;
xx=gamma_KMeans{ind_K};
clear x y
for t=1:(length(xx)-1)
x{t}=xx(:,t);y{t}=xx(:,t+1);
end
net = patternnet(10);
net.trainParam.showWindow = 0;
net = train(net,xx(:,1:(length(xx)-1)),xx(:,2:(length(xx))),'useParallel','yes');
%view(net)
yy = net(xx);
perf = perform(net,y,yy);
classes = vec2ind(yy);