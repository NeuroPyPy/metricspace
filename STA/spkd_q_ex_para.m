function d=spkd_q_ex_para(tli,tlj,costs)
%
% d=spkd(tli,tlj,costs) calculates the "spike time" distance
% (Victor & Purpura 1996) for a single costs
%
% tli: vector of spike times for first spike train
% tlj: vector of spike times for second spike train
% costs: costs per unit time to move a spike
%
% Copyright (c) 1999 by Daniel Reich and Jonathan Victor.
% Translated to Matlab by Daniel Reich from FORTRAN code by Jonathan Victor.
%%
% tli = (cat_ts{1});
% tlj = (cat_ts{2});
%%
% nspi=int32(size(tli,2));
% nspj=int32(size(tlj,2));
% nexch = size(tli,1);
% if costs==0
%     d=single(abs(nspi-nspj));
%     return
% elseif costs==Inf
%     d=single(nspi+nspj);
%     return
% end
% scr=zeros(nspi+1,nspj+1);
% scr(:,1)=(0:nspi)';
% scr(1,:)=(0:nspj);
% scr=single(repmat(shiftdim(scr,-1),[length(costs),1,1,nexch]));
% % tlsub = costs(:).*shiftdim(abs(tli-permute(tlj,[1 3 2])),-1);
% tli = permute(tli,[4 2 3 1]);
% tlj = permute(tlj,[4 2 3 1]);
% if nspi && nspj
%     for i=2:nspi+1
%         for j=2:nspj+1
%             scr(:,i,j,:)=min(cat(3,scr(:,i-1,j,:)+1,scr(:,i,j-1,:)+1,scr(:,i-1,j-1,:)+costs'.*abs(tli(:,i-1,:,:)-tlj(:,j-1,:,:))),[],3);
%         end
%     end
% end
% d=squeeze(scr(:,nspi+1,nspj+1,:));
%%
nspi=int32(size(tli,2));
nspj=int32(size(tlj,2));
nexch = size(tli,1);
if costs==0
    d=single(abs(nspi-nspj));
    return
elseif costs==Inf
    d=single(nspi+nspj);
    return
end
scr=zeros(nspi+1,nspj+1);
scr(:,1)=(0:nspi)';
scr(1,:)=(0:nspj);
scr=single(repmat(shiftdim(scr,-1),[length(costs),1,1,nexch]));
costs = costs.';
tli = permute(tli,[4 2 3 1]);
tlj = permute(tlj,[4 3 2 1]);
tlsub = tli - tlj;
% tlsub = tlsub.*permute(tli,[3 2 4 1]);
% tlsub = tlsub.*tli;
% tlsub = permute(tlsub,[2 3 4 1]);
% tlsub = permute(costs.*permute(abs(tli-permute(tlj,[1 3 2])),-1),[1 3 4 2]);
if nspi && nspj
    for i=2:nspi+1
        for j=2:nspj+1
            scr(:,i,j,:)=min(cat(3,scr(:,i-1,j,:)+1,scr(:,i,j-1,:)+1,scr(:,i-1,j-1,:)+costs.*tlsub(:,i-1,j-1,:)),[],3);
        end
    end
end
d=squeeze(scr(:,nspi+1,nspj+1,:));