function count_vector = count_successive_elements(arr)
    count_vector = [];

    i = 1;
    while i <= length(arr)
        current_element = arr{i};
        count = 0;
        while i <= length(arr) && strcmp(current_element, arr{i})
            count = count + 1;
            i = i + 1;
        end
        count_vector = [count_vector, count];
    end
end