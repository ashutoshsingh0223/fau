function [Boundary,data,label,label_ind] = BoundaryLabel_ij(label,data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[NX,NY,NZ]=size(label);
label_ind=[];
mask=ones(3,3,3);
Boundary=0;
k=1;
for i=1:6
        i
        xxx=(label(2:NX-1,2:NY-1,2:NZ-1)==i);
        bini=convn(double(label==i),mask,'valid')/27;%bin=bin(2:NX+1,2:NY+1,2:NZ+1);
        flagi{i}=((bini<26/27)&(bini>(1/27)));
        if sum(sum(sum(xxx)))>0
            Boundary=(Boundary-Boundary.*xxx)+k*xxx;
            label_ind=[label_ind;[i i]];
            k=k+1;
        end                
end

for i=1:6
    for j=(i+1):6
        disp(['Checking Interface ' num2str(i) ', ' num2str(j)])
        flag=double((flagi{i})&(flagi{j}));
        if sum(sum(sum(flag)))>0
            Boundary=(Boundary-Boundary.*flag)+(k)*flag;
            label_ind=[label_ind;[i j]];
            k=k+1;
        end
        %Label_filt=flag*i;
    end
end
data=data(2:NX-1,2:NY-1,2:NZ-1);
label=label(2:NX-1,2:NY-1,2:NZ-1);
end

