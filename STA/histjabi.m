function h=histjabi(cvec)
% h=histjabi(cvec) gives the jackknife bias that should be added
% to the naive plugin estimate, in bits
%
% cvec is the vector of counts.  See entest.doc (JV)
%
% See also HISTTPBI, JACKENT.
%
cvec=reshape(cvec,prod(size(cvec)),1);
if (max(cvec)<=1) h=0; return; end
nsamps=sum(cvec);
cv2=cvec(find(cvec>=2));
jdev=log((nsamps-1)/nsamps)+(1/nsamps/(nsamps-1))*sum(cv2.*(cv2-1).*log(cv2./(cv2-1)));
h=-(nsamps-1)*jdev/log(2);  % first-order correction
