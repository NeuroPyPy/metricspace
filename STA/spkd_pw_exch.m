function [ D_exch ] = spkd_pw_exch( cspks, qvals, idx, nexch )

nq = numel(qvals);
[idx, idxind] = sort(idx);
cspks = cspks(idxind);
curcounts = cellfun(@numel, cspks);
uidx = unique(idx);
cat_ts = cell(nexch,1);
numt = numel(cspks);

for xex = 1:nexch
    cat_ts{xex} = cspks;
    for xid = 1:numel(uidx)
        allts = cat(2,cspks{idx==uidx(xid)});
        cat_ts{xex}(idx==uidx(xid)) = cellfun(@sort, mat2cell(allts(randperm(numel(allts),numel(allts))).',curcounts(idx==uidx(xid))), 'UniformOutput', false);
    end
    cat_ts{xex} = cellfun(@transpose, cat_ts{xex}, 'UniformOutput', false);
end

cat_ts = cellfun(@(X) cat(1,X{:}), num2cell(cat(2,cat_ts{:}),2), 'UniformOutput',false);

SD1 = cellfun(@(X) permute(X,[3 2 1]), cat_ts, 'UniformOutput', false);
SD2 = cellfun(@(X) permute(X,[2 3 1]), cat_ts, 'UniformOutput', false);

D_exch = zeros(numt,numt,nq,nexch);
go = curcounts~=0;
tic
for xi = 1:(numt-1)
    for xj = (xi+1):numt 
        if go(xi) && go(xj)
            SD = permute(abs(SD1{xi}-SD2{xj}).*permute(qvals(:),[2 3 4 1]),[4 2 1 3]);
            scr = repmat(shiftdim(zeros(curcounts(xi)+1,curcounts(xj)+1,'single'),-1),[nq,1,1,nexch]);
            scr(:,:,1,:) = scr(:,:,1,:)+shiftdim((0:curcounts(xi)).',-1);
            scr(:,1,:,:) = scr(:,1,:,:)+shiftdim((0:curcounts(xj)),-1);
            D_exch(xi,xj,:,:) = spkd_v_exch_mex(scr,SD,qvals,nexch);
        else
            D_exch(xi,xj,:,:) = max(curcounts([xi xj]));
        end
    end
end

D_exch = max(D_exch,permute(D_exch,[2 1 3 4]));

end
