function [ Ng,L,T ] = get_size_of_ppi( ppi )
% if someone is too lazy to write size(...)
% and the check of consistency of input variables
%

Ng = length(ppi);
L = size(ppi{1},1);
T = zeros(1,Ng);
for g=1:Ng
   T(g) = size(ppi{g},2);
    
   % check the consistency in given ppi
   % (K of all ppi{g} has to be the same)
   if size(ppi{g},1) ~= L
       error('ppi: inconsistent data provided')
   end
end

end

