function [ Ng,n,T ] = get_size_of_x( x )
% if someone is too lazy to write size(...)
% and the check of consistency of input variables
%

Ng = length(x);
n = size(x{1},1);
T = zeros(1,Ng);
for g=1:Ng
   T(g) = size(x{g},2);
    
   % check the consistency in given ppi
   % (K of all ppi{g} has to be the same)
   if size(x{g},1) ~= n
       error('x: inconsistent data provided')
   end
end

end

