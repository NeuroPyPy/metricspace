function [ minD, minOffset ] = spkd_slide( cspks, minOffset, maxOffset, resolution )

    % create array of offset values with a resolution (maybe default to 1ms)
    offsetRange = minOffset:resolution:maxOffset;
    numOffsets = numel(offsetRange);
    
    % preallocate because thats what we do in matlab
    D = zeros(numOffsets, 1);

    % implement the exhaustive search
    for oi = 1:numOffsets
        offset = offsetRange(oi);
        
        % cspks is the original input to spkd.m, so here apply the offset
        cspksOffset = cellfun(@(spk) spk + offset, cspks, 'UniformOutput', false);

        % compute distance using original function
        D(oi) = spkd_pw_py(cspksOffset);
    end

    % find minimum distance and corresponding offset
    [minD, minIndex] = min(D);
    minOffset = offsetRange(minIndex);

end
