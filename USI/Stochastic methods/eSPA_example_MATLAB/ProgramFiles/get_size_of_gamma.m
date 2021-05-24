function [ Ng,K,T ] = get_size_of_gamma( gamma )
% if someone is too lazy to write size(...)
% and the check of consistency of input variables
%

Ng = length(gamma);
K = size(gamma{1},1);
T = zeros(1,Ng);
for g=1:Ng
   T(g) = size(gamma{g},2);
    
   % check the consistency in given gamma
   % (K of all gamma{g} has to be the same)
   if size(gamma{g},1) ~= K
       error('gamma: inconsistent data provided')
   end
end

end

