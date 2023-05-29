function h=tblxbi(ctabl,type,param)

% function h=tblxbi(ctabl,type,param) gives the Treves-Panzeri or jacknife bias correction
% to be added to the naive transinformation estimate, in bits
%
% ctabl must be 2-dimensional, and contains the counts.
%   sum(ctabl) is the total number of trials
% type (first two characters, made lower) determines kind of estimate
% type='ja', 'Jackknife', etc.  for jackknife estimate
% type='tr', 'Treves-Panzeri', etc., for Treves-Panzeri estimate
% param:
%   for type='tr', 'Treves-Panzeri', etc., this is "useall",
%      useall=1 to use all bins in the table
%      useall=0 (default) to use bins that are in an occupied row or occupied column
%      useall=-1 to just use nonzero bins
%   for type='ja', ignored
%
%
% See also TBLXINFO, HISTTPBI, HISTJABI, HISTBI.
%
ty=lower(type(1:2));
switch ty
	case 'ja'
      h=histbi(sum(ctabl,1),'ja')+histbi(sum(ctabl,2),'ja')-histbi(ctabl,'ja');
   case 'tr'
		if (nargin <=2); useall=0; else useall=param; end
      h=tblxtpbi(ctabl,useall);
end
return
