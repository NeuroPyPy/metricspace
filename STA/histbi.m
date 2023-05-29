function h=histbi(cvec,type,param)
% h=histbi(cvec,type,param) gives 
% a bias estimate for the naive plugin histogram entropy estimate, in bits
%
% cvec is the vector of counts
% type='ja' for jackknife estimate
% type='tp' for Treves-Panzeri estimate
% type='cs' for Chao-Shen estimate
% param:
%   for type='tp', this is "useall",
%       useall=0 (default) to just count the nonzero bins in pvec; 1 to use all bins
%   for type='ja', ignored
%
% See also HISTTPBI, HISTJABI.
%
switch type
    case 'ja'
        h=histjabi(cvec);
    case 'tp'
        if (nargin <=2); useall=0; else useall=param; end
        h=histtpbi(cvec,useall);
    case 'cs'
        h=histcsbi(cvec,'least');
end
return

