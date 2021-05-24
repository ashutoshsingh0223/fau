function [out] = EraseRedundantClusters(out,i,j)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[~,ii]=find(sum(out{i}.gamma{j}')>1e-7);
out{i}.gamma{j}=out{i}.gamma{j}(ii,:);
out{i}.C{j}=out{i}.C{j}(:,ii);
out{i}.Lambda{j}=out{i}.Lambda{j}(:,ii);

end

