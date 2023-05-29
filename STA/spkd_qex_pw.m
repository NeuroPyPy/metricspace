function D_exch = spkd_qex_pw(cat_ts,curcounts,numt,qvals,nexch)
%%


D_exch = zeros(numt,numt,numel(qvals),nexch);
for xi = 1:(numt-1)
    if curcounts(xi)==0
        D_exch(xi,:,:,:) = repmat(curcounts(:).',1,1,numel(qvals),nexch);
    else
        for xj = (xi+1):numt
            if curcounts(xj)~=0
                D_exch(xi,xj,:,:) = spkd_q_ex_para(cat_ts{xi},cat_ts{xj},qvals);
            end
        end
    end
end

D_exch = max(D_exch, permute(D_exch,[2 1 3 4]));
%%
end