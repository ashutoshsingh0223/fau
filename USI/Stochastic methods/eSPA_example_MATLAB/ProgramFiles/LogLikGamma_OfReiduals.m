function LogLik = LogLikGamma_OfReiduals(x)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[phat,pci] = mle(x,'distribution','gamma');
LogLik = sum(log(gampdf(x,phat(1),phat(2))));
end

