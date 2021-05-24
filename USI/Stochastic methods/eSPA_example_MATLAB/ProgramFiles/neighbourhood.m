function X = neighbourhood(S,ind_frame_x,ind_frame_y,t)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
X=zeros(length(ind_frame_x)-2,length(ind_frame_y)-2,7);
n_x=1;
for x=ind_frame_x(2:(length(ind_frame_x)-1))
    n_y=1;
    for y=ind_frame_y(2:(length(ind_frame_y)-1))
             X(n_x,n_y,1)=S(x,y,t);
             X(n_x,n_y,2)=S(x,y,t-1);
             X(n_x,n_y,3)=S(x,y,t+1);
             X(n_x,n_y,4)=S(x-1,y,t);
             X(n_x,n_y,5)=S(x+1,y,t);
             X(n_x,n_y,6)=S(x,y+1,t);
             X(n_x,n_y,7)=S(x,y-1,t);
             n_y=n_y+1;
    end
    n_x=n_x+1;
end

