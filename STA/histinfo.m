function h=histinfo(pvec)
%
% function h=histinfo(pvec) gives the histogram information, in bits
% sum(pvec) assumed to be 1, and all elements positive, but needn't be 1-dimensional.
% lacks all the bells and whistles of histent.
%
% See also TBLXINFO, HISTENT.
%
pnz=reshape(pvec,1,prod(size(pvec)));
pnz=pnz(find(pnz>0));
if (isempty(pnz))
   h=0;
else
   h=-pnz*log(pnz)'/log(2);
end
return

