%% Computation of the Gini-index for the 
%% data matrix X
%%
%% (C) Illia Horenko, 2018

function G = MyGini(X)
[n,T]=size(X);
G=zeros(n,1);
for i=1:n
    ss=sum(X(i,:));
    if ss>1
    for t1=1:T
        for t2=1:T
            G(i)=G(i)+abs(X(i,t1)-X(i,t2));
        end
    end
    G(i)=G(i)/(2*ss*T);
    end
end
end

