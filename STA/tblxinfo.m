function h=tblxinfo(tabl)
%
% h=tblxinfo(tabl) gives the transinformation, in bits
% tabl must be 2-dimensional.
%
% sum(tabl) assumed to be 1, and all elements positive.
%
% See also HISTINFO, HISTENT.
%
h=histinfo(sum(tabl,1))+histinfo(sum(tabl,2))-histinfo(tabl);
return

