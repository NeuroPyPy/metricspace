function [ labels ] = metricPredict( D, idx )
%% Get predicted labels from distance matrices, allowing ties
% INPUTS:
%    D:
%     N x N x ? array.  Each 'page' is an NxN distance matrix, where N is
%     the total number of trials.  May have any number of additional
%     dimensions, which may correspond to q values, t values, or exchange
%     resamplings.
%
%    idx:
%     True class labels, as row or column vector.
%
% OUTPUT:
%    labels:
%     N x K x ? array, where K is the number of classes.  As with D, may
%     have an unlimited number of 'pages' indexed in dimensions 3 and
%     above.  Each N x K array indicates the predicted class assignment of
%     all N trials, with 'soft' assignments allowed in the case of ties.
%     E.g., the i-th row of the j-th column is the predicted likelihood
%     that trial #i belongs to class j.
%
%%
% D=D1(tind,tind,10,1);
expo = -2;

Dz = D<eps;
D = D + diag(NaN(size(D,1),1));
Dz = Dz + diag(NaN(size(Dz,1),1));

classmat = single(idx.' == unique(idx));
classmat(classmat==0) = NaN;

avd = squeeze(mean((shiftdim(D,-1).*classmat).^expo,2,'omitnan')).^(1/expo); % mean class distances

% if ztrump
    zfrc = -squeeze(mean((shiftdim(Dz,-1).*classmat),2,'omitnan')); % mean class distances
    avd(logical(zfrc)) = zfrc(logical(zfrc));
% end





labels = avd==min(avd,[],1); % Identify all classes that have minimum distance with each trial
labels = labels./sum(labels,1); % Creates fractional counts for distance ties

labelDims = 1:numel(size(labels));
labelDims([1 2]) = labelDims([2 1]);
labels = single(permute(labels,labelDims));

%%
end
% Alex Denman (Alex.Denman.Brice@gmail.com)