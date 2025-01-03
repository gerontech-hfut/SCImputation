function imputed2 = SCPKNN_RBF_kernel(missing, sigma, K_proportion)
    % 设置默认参数值
    if nargin < 3
        K_proportion = 0.01;
        if nargin < 2
            sigma = 0.008;
        end
    end

    complete = [];
    no_complete = [];
    no_complete_index = [];
    complete_index = [];
    [row, col] = size(missing);
    Mi_dataset = zeros(col, col);
    gauss_sum = zeros(col);
    gauss_dataset = zeros(col, col);
    logic_dataset = zeros(col, col);
    total_K = 0;
    total_count = 0; 

    for i = 1:row
        if isempty(find(missing(i, :) == 9, 1))
            continue;
        else
            no_complete_index = [no_complete_index, i];
        end
    end
    no_complete = missing(no_complete_index, :);
    complete_index = setdiff(1:row, no_complete_index);
    complete = missing(complete_index, :);

    for k = 1:col
        sum_i = 0;
        for j = 1:col
            if k ~= j
                sum_i = sum_i + abs(corr(missing(:, j), missing(:, k)));
            end
        end
        for j = 1:col
            if k ~= j
                Mi_dataset(k, j) = abs(corr(missing(:, j), missing(:, k)) / sum_i);
            end
        end
        for j = 1:col
            Mi_dataset(j, j) = max(Mi_dataset(j, :));
        end
    end

    for k = 1:col
        gauss_sum(k) = 0;
        for j = 1:col
            gauss_dataset(k, j) = RBF_function(Mi_dataset(k, k), Mi_dataset(k, j), sigma);
            gauss_sum(k) = gauss_sum(k) + gauss_dataset(k, j);
        end
    end
    RATE_DE = 5;
    gauss_sum = round((gauss_sum / max(gauss_sum)) * col + col / RATE_DE);
    gauss_sum(gauss_sum > col) = col;
    Mi_dataset = zeros(col, col);
    for k = 1:col    
        real_index = MAX_K_NUMBER(gauss_dataset(k, :), gauss_sum(k));
        Mi_dataset(k, real_index) = gauss_dataset(k, real_index);
    end

    %imputed1=Mi_dataset;

    missing_rate = [];
    for no_com = 1:length(no_complete_index)
        nan_index = find(no_complete(no_com, :) == 9);
        nan_num = length(nan_index);
        nan_rate = nan_num / col;
        missing_rate = [missing_rate, nan_rate];
    end
    [~, rate_rank] = sort(missing_rate);

    % missing data imputation
    for i = 1:length(rate_rank)
        if ~any(~(no_complete(rate_rank(i), :) == 9))
            continue;
        end
        miss_row_index = rate_rank(i);
        miss_atr_index = find(no_complete(rate_rank(i), :) == 9);
        miss_atr_num = length(miss_atr_index);
        %weight_array = [];
        column_index1 = find(no_complete(rate_rank(i), :) ~= 9);

        %op_column_index = ~column_index;
        for m = 1:miss_atr_num
            column_index = intersect(column_index1, find(Mi_dataset(miss_atr_index(m), :) ~= 0));
            temp = repmat(no_complete(rate_rank(i), column_index), length(complete_index), 1);
            distance = power((temp - complete(:, column_index)), 2);
            distance = distance * (Mi_dataset(miss_atr_index(m), column_index))';
            distance = sqrt(distance);
            %[~, distance_index] = sort(distance, 'descend');
            [~, distance_index] = sort(distance);
            make_K = sum(distance) * K_proportion;
            sum_distance = 0;
            K = 1;
            while(sum_distance < make_K)
                sum_distance = sum_distance + distance(distance_index(K));
                K = K + 1;
            end

            total_K = total_K + K;
            total_count = total_count + 1;
            adjusted_distance = distance(distance_index(1:K)) + 1e-10;

            distance_p = 1 ./ adjusted_distance;
            distance_p = distance_p / sum(distance_p); 

            neighbor_values = complete(distance_index(1:K), miss_atr_index(m));
            unique_neighbor_values = unique(neighbor_values);
            value_counts = zeros(size(unique_neighbor_values));
            for uv = 1:length(unique_neighbor_values)
                value_counts(uv) = sum(complete(:, miss_atr_index(m)) == unique_neighbor_values(uv));
            end
            value_probs = value_counts / sum(value_counts);

            weighted_values = zeros(size(unique_neighbor_values));
            for ka = 1:K
                sample_value = neighbor_values(ka);
                value_index = find(unique_neighbor_values == sample_value);
                weighted_values(value_index) = weighted_values(value_index) + distance_p(ka) * length(unique_neighbor_values) * value_probs(value_index);
            end

            [~, max_index] = max(weighted_values);
            impute_value = unique_neighbor_values(max_index);      

            %[~, max_index] = max(weighted_values);
            %impute_value = unique_neighbor_values(max_index);
            no_complete(rate_rank(i), miss_atr_index(m)) = impute_value;
        end
    end
    missing(no_complete_index, :) = no_complete;
    imputed2 = missing;

    average_K = total_K / total_count;
    fprintf('Average K: %f\n', average_K);
end
