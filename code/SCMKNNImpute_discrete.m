function imputed2 = SCMKNNImpute_discrete(missing, sigma, K_proportion)
% SCMKNNImpute_discrete Impute missing discrete data using Structural Causal Model with K-Nearest Neighbors.
%
%   imputed2 = SCMKNNImpute_discrete(missing)
%   imputed2 = SCMKNNImpute_discrete(missing, sigma)
%   imputed2 = SCMKNNImpute_discrete(missing, sigma, K_proportion)
%
%   Inputs:
%       missing        - [p x q] dataset with missing values denoted by 9
%       sigma          - [optional] RBF kernel parameter (default: 0.08)
%       K_proportion   - [optional] Proportion for dynamic neighbor selection (default: 0.01)
%
%   Output:
%       imputed2       - [p x q] imputed dataset

    % 设置默认参数值
    if nargin < 3
        K_proportion = 0.01;
        if nargin < 2
            sigma = 0.008;
        end
    end

    % 初始化变量
    complete = [];
    no_complete = [];
    no_complete_index = [];
    complete_index = [];
    [row, col] = size(missing);
    Mi_dataset = zeros(col, col);
    gauss_sum = zeros(col, 1);
    gauss_dataset = zeros(col, col);
    % logic_dataset 在此函数中未使用，故注释掉
    % logic_dataset = zeros(col, col);   
    
    % 提取完整数据和不完整数据
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
    
    % 计算相关性矩阵，使用 Kendall 相关系数
    for k = 1:col
        sum_i = 0;
        for j = 1:col
            if k ~= j
                sum_i = sum_i + abs(corr(missing(:, j), missing(:, k), 'type', 'Kendall'));
            end
        end
        for j = 1:col
            if k ~= j
                % 修正语法错误：将 'type', 'Kendall' 作为 corr 函数的参数
                Mi_corr = corr(missing(:, j), missing(:, k), 'type', 'Kendall');
                Mi_dataset(k, j) = abs(Mi_corr / sum_i);
            else
                Mi_dataset(k, j) = max(Mi_dataset(j, :)); 
            end
        end
        % 确保对角线元素为各行的最大值
        for j = 1:col
            Mi_dataset(j, j) = max(Mi_dataset(j, :));
        end
    end
    
    % 应用 RBF 核函数到 MI 矩阵以创建权重矩阵
    for k = 1:col
        gauss_sum(k) = 0;
        for j = 1:col
            gauss_dataset(k, j) = RBF_function(Mi_dataset(k, k), Mi_dataset(k, j), sigma);
            gauss_sum(k) = gauss_sum(k) + gauss_dataset(k, j);
        end
    end
    
    % 归一化 Gaussian 和并限制在特征数量范围内
    gauss_sum = round((gauss_sum / max(gauss_sum)) * col + col);
    gauss_sum(gauss_sum > col) = col;
    Mi_dataset = zeros(col, col);
    
    % 基于归一化的 Gaussian 权重选择前 K 个邻居
    for k = 1:col    
        real_index = MAX_K_NUMBER(gauss_dataset(k, :), gauss_sum(k));
        Mi_dataset(k, real_index) = gauss_dataset(k, real_index);
    end
        
    imputed1 = Mi_dataset;
    
    % 计算缺失率
    missing_rate = [];
    for no_com = 1:length(no_complete_index)
        nan_index = find(no_complete(no_com, :) == 9);
        nan_num = length(nan_index);
        nan_rate = nan_num / col;
        missing_rate = [missing_rate, nan_rate];
    end
    [~, rate_rank] = sort(missing_rate);
    SIGMA_DE2 = 2;
    
    % 缺失数据填补
    for i = 1:length(rate_rank)
        if ~any(~(no_complete(rate_rank(i), :) == 9))
            continue;
        end
        miss_row_index = rate_rank(i);
        miss_atr_index = find(no_complete(rate_rank(i), :) == 9);
        miss_atr_num = length(miss_atr_index);
        % weight_array = []; % 未使用
        column_index1 = find(no_complete(rate_rank(i), :) ~= 9);%
        
        % 对每个缺失属性进行填补
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
            while(sum_distance < make_K && K <= length(distance))
                sum_distance = sum_distance + distance(distance_index(K));
                K = K + 1;
            end
            
            % 修改距离以确保没有零值
            adjusted_distance = distance(distance_index(1:K)) + 1e-10;
            distance_p = 1 ./ adjusted_distance;
            distance_p = distance_p / sum(distance_p);

            % 计算 P(F = f_target)
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
end