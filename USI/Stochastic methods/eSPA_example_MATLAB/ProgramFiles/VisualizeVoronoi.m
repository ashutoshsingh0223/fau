function []=VisualizeVoronoi(C,P)

C=C(1:2,:);
%C=D(:,size(D,1)-1:size(D,1))'*Centroids;
figure;hold on;
N_grid=100;delta=0.1;
[XX,YY]=meshgrid((min(C(1,:))-delta):(max(C(1,:))-min(C(1,:))+2*delta)/N_grid:max(1,(max(C(1,:))+delta)),(min(C(2,:))-delta):(max(C(2,:))-min(C(2,:))+2*delta)/N_grid:max((max(C(2,:))+delta),1));
xx=reshape(XX,1,numel(XX));yy=reshape(YY,1,numel(YY));
[~,idx] = min(sqDistance([xx; yy;repmat(mean(C(3:size(C,1),:)')',1,length(xx))], C)');
zz=0*xx;
for t=1:length(idx)
zz(t)=P(idx(t));%round(P(idx(t))*100)/100;
end
contourf(XX,YY,reshape(zz,size(XX,1),size(XX,2)),P);alpha 0.5
h = voronoi(C(1,:)',C(2,:)');hold on;
for t=1:size(C,2)
text(C(1,t)+0.005,C(2,t)+0.005,[num2str(t)],'FontSize',24);
end
h(1).MarkerSize=10;h(1).Marker='o';h(1).MarkerFaceColor='k';
set(gcf,'Position',[10 100 800  600]);
set(gca,'FontSize',24,'LineWidth',2);
axis off
colormap jet
caxis([0 0.9])
