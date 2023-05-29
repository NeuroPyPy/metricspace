function [ D_exch ] = spkd_v_exch( scr, SD, qvals, nexch )

%%

D_exch = zeros(numel(qvals),nexch,'single');


for xii=2:(size(SD,2)+1)
    for xjj=2:(size(SD,3)+1)
        scr(:,xii,xjj,:)= ...
            min(cat(3, ...
            scr(:,xii-1,xjj,:)+1, ...
            scr(:,xii,xjj-1,:)+1, ...
            ( ...
                scr(:,xii-1,xjj-1,:) ...
                + ...
                SD(:,xii-1,xjj-1,:) ...
            )...
            ),[],3);
    end
end
D_exch(:,:) = squeeze(scr(:,end,end,:));


%%
end

% function [ D_exch ] = spkd_v_exch( DD, SD, numt, curcounts, qvals, nexch )
% 
% %%
% go = curcounts~=0;
% 
% D_exch = zeros(numt,numt,numel(qvals),nexch,'single');
% 
% 
% for xi = 1:(numt-1)
% %     if go(xi)
%         for xj = (xi+1):numt
%             if go(xi)&&go(xj)
%                 for xii=2:(curcounts(xi)+1)
%                     for xjj=2:(curcounts(xj)+1)
%                         DD{xi,xj}(:,xii,xjj,:)= ...
%                             min(cat(3, ...
%                             DD{xi,xj}(:,xii-1,xjj,:)+1, ...
%                             DD{xi,xj}(:,xii,xjj-1,:)+1, ...
%                             ( ...
%                                 DD{xi,xj}(:,xii-1,xjj-1,:) ...
%                                 + ...
%                                 SD{xi,xj}(:,xii-1,xjj-1,:) ...
%                             )...
%                             ),[],3);
%                     end
%                 end
%                 D_exch(xi,xj,:,:) = squeeze(DD{xi,xj}(:,end,end,:));
%             else
%                 D_exch(xi,xj,:,:) = max(curcounts([xi xj]));
%             end
%         end
% %     else
% %         D_exch(xi,:,:,:) = repmat(curcounts(:).',[1,1,numel(qvals),nexch]);
% %     end
% end
% 
% %%
% end
