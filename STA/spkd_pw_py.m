function [ D ] = spkd_pw_py( cspks )

qvals = double([0 2.^(-4:0.5:9)]);
curcounts = cellfun(@numel, cspks);
numt = size(cspks, 1);

if numt < 1 % debug, should not happen
    D = ones(10,2,numel(qvals),'double');
    disp("numt < 1")
    return;
end

[a, ~] = size(curcounts);
if a == 1
    disp("spike counts == 1; transposing...")
    curcounts = curcounts.';
    [newa, ~] = size(curcounts);
    if newa == 1
        disp("numt still == 1 after transposition")
        return;
    end
end

if numt == 1 % debug, transpose from python list of 1xN
    disp("numt == 1; transposing...")
    cspks = cspks.';
    numt = size(cspks, 1);
    if numt == 1
        disp("numt still == 1 after transposition")
        D = ones(5, 5, 5, 'double'); % debug return for distinguishing different numt's
        return
    end
end
% Process spikes for pairwise distance calculation 
D = zeros(numt,numt,numel(qvals),'double');
for xi = 1:(numt-1)
    for xj = (xi+1):numt 
        if curcounts(xi)~=0 && curcounts(xj) ~=0
            SD = permute(abs(cspks{xi}-cspks{xj}.').*permute(qvals(:),[2 3 1]),[3 2 1]);
            
            scr = repmat(shiftdim(zeros(curcounts(xi)+1,curcounts(xj)+1,'double'),-1),[numel(qvals),1,1]);
            scr(:,:,1) = scr(:,:,1)+shiftdim((0:curcounts(xi)).',-1);
            scr(:,1,:) = scr(:,1,:)+shiftdim((0:curcounts(xj)),-1);
            D(xi,xj,:) = spkd_v(scr,(SD),(qvals));
        else
            D(xi,xj,:) = max(curcounts([xi xj]));
        end
    end
end
D = max(D,permute(D,[2 1 3]));
end
