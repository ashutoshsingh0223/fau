function [X,pi]=generate_challenge1(T,sigma,d);
s=0.1*sigma; mu=4;
X=[mvnrnd([-mu zeros(1,1)],diag([s sigma*ones(1,1)]),T)' mvnrnd([0 zeros(1,1)],diag([s sigma*ones(1,1)]),T)' mvnrnd([mu zeros(1,1)],diag([s sigma*ones(1,1)]),T)'];
X=[X;rand(d-1,size(X,2))];

alpha=45*3.1415/180;

X(1:2,:)=[cos(alpha) -sin(alpha);sin(alpha) cos(alpha)]*X(1:2,:);
for i=1:size(X,1)
   X(i,:)=X(i,:)-mean(X(i,:));
   X(i,:)=X(i,:)/max(abs(X(i,:)));
end

pi(1,:)=[ones(1,T) zeros(1,T) ones(1,T)];
pi(2,:)=1-pi(1,:);