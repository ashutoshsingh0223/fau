function Y=CoarseGrainDim(X,range_dim,n)
WS=floor((range_dim(2)-range_dim(1)+1)/n);
win=range_dim(1):WS:range_dim(2);
if win(length(win))<range_dim(2)
    win=[win range_dim(2)+1];
end
Y=zeros(length(win)-1,size(X,2));
for i=1:(length(win)-1)
    Y(i,:)=mean(X(win(i):(win(i+1)-1),:));
end

Y=[X(setdiff(1:size(X,1),range_dim(1):range_dim(2)),:);Y];