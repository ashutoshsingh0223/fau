function [X,pi]=generate_challenge2(T,sigma,d);

s=0.5;
phi=2*3.141516*[rand(1,2*T)];rad=[0+s*randn(1,T) 5+s*randn(1,T)];

%alpha=3.1415/4;

X(1:2,:)=[rad.*sin(phi) 6+rad.*sin(phi);rad.*cos(phi) 2+rad.*cos(phi)];
if d>2
X(3:(d+1),:)=rand(d-1,size(X,2));
end

for i=1:size(X,1)
   X(i,:)=X(i,:)-mean(X(i,:));
   X(i,:)=X(i,:)/max(abs(X(i,:)));
end

pi(1,:)=[ones(1,T) zeros(1,T) ones(1,T) zeros(1,T)];
pi(2,:)=1-pi(1,:);