function imputed2 = SCPKNNImpute_discrete(missing, sigma, K_proportion)
% SCPKNNImpute_discrete Impute missing discrete data using Structural Causal Model with K-Nearest Neighbors.
%
%   imputed2 = SCPKNNImpute_discrete(missing)
%   imputed2 = SCPKNNImpute_discrete(missing, sigma)
%   imputed2 = SCPKNNImpute_discrete(missing, sigma, K_proportion)
%
%   Inputs:
%       missing        - [p x q] dataset with missing values denoted by 9
%       sigma          - [optional] RBF kernel parameter (default: 0.005)
%       K_proportion   - [optional] Proportion for dynamic neighbor selection (default: 0.01)
%
%   Output:
%       imputed2       - [p x q] imputed dataset

    % 设置默认参数值
    if nargin < 3
        K_proportion = 0.01;
        if nargin < 2
            sigma = 0.005;
        end
    end

    % 参数验证
    if sigma <= 0
        error('Sigma 必须为正数。');
    end
    if K_proportion <= 0 || K_proportion > 1
        error('K_proportion 必须在 (0, 1] 之间。');
    end

    % 初始化变量
    [row, col] = size(missing);
    
    % 提取完整数据和不完整数据
    incomplete_mask = any(missing == 9, 2);
    no_complete_index = find(incomplete_mask);
    complete_index = find(~incomplete_mask);
    no_complete = missing(no_complete_index, :);
    complete = missing(complete_index, :);
    
    % 计算相关性矩阵，使用 Kendall 相关系数
    Mi_dataset = zeros(col, col);
    for k = 1:col
        sum_i = 0;
        for j = 1:col
            if k ~= j
                current_corr = corr(missing(:, j), missing(:, k), 'type', 'Kendall');
                sum_i = sum_i + abs(current_corr);
            end
        end
        for j = 1:col
            if k ~= j
                current_corr = corr(missing(:, j), missing(:, k), 'type', 'Kendall');
                Mi_dataset(k, j) = abs(current_corr / sum_i);
            else
                Mi_dataset(k, j) = max(Mi_dataset(j, :)); 
            end
        end
    end
    
    % 确保对角线元素为各行的最大值
    for j = 1:col
        Mi_dataset(j, j) = max(Mi_dataset(j, :));
    end

    % 应用 RBF 核函数到 MI 矩阵以创建权重矩阵
    gauss_dataset = RBF_function_matrix(Mi_dataset, sigma);
    gauss_sum = sum(gauss_dataset, 2);
    
    % 归一化 Gaussian 和并限制在特征数量范围内
    gauss_sum = round((gauss_sum / max(gauss_sum)) * col + col);
    gauss_sum(gauss_sum > col) = col;
    
    % 选择前 K 个邻居
    Mi_dataset = zeros(col, col);
    for k = 1:col    
        real_index = MAX_K_NUMBER(gauss_dataset(k, :), gauss_sum(k));
        Mi_dataset(k, real_index) = gauss_dataset(k, real_index);
    end
    
    % 计算缺失率
    missing_rate = sum(no_complete == 9, 2) / col;
    [~, rate_rank] = sort(missing_rate);
    
    SIGMA_DE2 = 2;

    % 缺失数据填补
    for i = 1:length(rate_rank)
        current_row = rate_rank(i);
        if all(no_complete(current_row, :) ~= 9)
            continue;
        end
        miss_atr_index = find(no_complete(current_row, :) == 9);
        miss_atr_num = length(miss_atr_index);
        column_index1 = find(no_complete(current_row, :) ~= 9);
        
        for m = 1:miss_atr_num
            % 找到相关列
            column_index = intersect(column_index1, find(Mi_dataset(miss_atr_index(m), :) ~= 0));
            if isempty(column_index)
                continue; % 如果没有相关列，则跳过
            end
            
            % 计算距离
            temp = repmat(no_complete(current_row, column_index), length(complete_index), 1);
            distance = sqrt(sum((temp - complete(:, column_index)).^2, 2));
            
            % 排序距离
            [~, distance_index] = sort(distance);
            
            % 确定 K 的数量
            make_K = sum(distance) * K_proportion;
            sum_distance = 0;
            K = 1;
            while sum_distance < make_K && K <= length(distance)
                sum_distance = sum_distance + distance(distance_index(K));
                K = K + 1;
            end
            K = min(K-1, length(distance)); % 确保 K 不超过实际距离数量
            
            % 获取邻居值
            neighbor_values = complete(distance_index(1:K), miss_atr_index(m));
            
            % 修改距离以避免除以零
            adjusted_distance = distance(distance_index(1:K)) + 1e-10;
            distance_p = 1 ./ adjusted_distance;
            distance_p = distance_p / sum(distance_p);
    
            % 计算 P(F = f_target)
            unique_neighbor_values = unique(neighbor_values);
            value_counts = histcounts(neighbor_values, [unique_neighbor_values; max(unique_neighbor_values)+1]);
            value_probs = value_counts / sum(value_counts);
            
            % 构建加权值
            weighted_values = zeros(size(unique_neighbor_values));
            for ka = 1:K
                sample_value = neighbor_values(ka);
                value_index = find(unique_neighbor_values == sample_value);
                if ~isempty(value_index)
                    weighted_values(value_index) = weighted_values(value_index) + distance_p(ka) * length(unique_neighbor_values) * value_probs(value_index);
                end
            end
    
            % 选择加权值最大的值作为填补值
            [~, max_index] = max(weighted_values);
            impute_value = unique_neighbor_values(max_index);      
            no_complete(current_row, miss_atr_index(m)) = impute_value;
        end
    end

    % 更新原始数据集中的缺失值
    missing(no_complete_index, :) = no_complete;
    imputed2 = missing;
end