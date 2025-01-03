function imputed2 = SCPLLSImpute_continuity(missing, sigma, K_proportion)
% SCPLLSImpute_continuity Impute missing continuous data using Structural Causal Model with PLLSimpute.
%
%   imputed2 = SCPLLSImpute_continuity(missing)
%   imputed2 = SCPLLSImpute_continuity(missing, sigma)
%   imputed2 = SCPLLSImpute_continuity(missing, sigma, K_proportion)
%
%   Inputs:
%       missing        - [p x q] dataset with missing values denoted by 9
%       sigma          - [optional] RBF kernel parameter (default: 0.06)
%       K_proportion   - [optional] Proportion for dynamic neighbor selection (default: 0.01)
%
%   Output:
%       imputed2       - [p x q] imputed dataset

    % 设置默认参数值
    if nargin < 3
        K_proportion = 0.01;
        if nargin < 2
            sigma = 0.06;
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
    
    % 计算相关性矩阵，使用 Pearson 相关系数
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
            else
                Mi_dataset(k, j) = max(Mi_dataset(j, :)); 
            end
        end
    end
    % Best RMSE: 0.271504, Best sigmaor: 0.040000, Best rateor: 1.000000, Best K_proportionor: 0.200000, Best sigma_2: 0.250000
    
    % 应用 RBF 核函数到 MI 矩阵以创建权重矩阵
    for k = 1:col
        gauss_sum(k) = 0;
        for j = 1:col
            gauss_dataset(k, j) = RBF_function(1, Mi_dataset(k, j), sigma);
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
        
    % 初始化 missing_rate（当前未使用）
    missing_rate = [];
    for no_com = 1:length(no_complete_index)
        nan_index = find(no_complete(no_com, :) == 9);
        nan_num = length(nan_index);
        nan_rate = nan_num / col;
        % missing_rate = [missing_rate, nan_rate]; % 如果需要使用，请取消注释
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
        column_index1 = find(no_complete(rate_rank(i), :) ~= 9);
        
        % 对每个缺失属性进行填补
        for m = 1:miss_atr_num
            column_index = intersect(column_index1, find(Mi_dataset(miss_atr_index(m), :) ~= 0));
            temp = repmat(no_complete(rate_rank(i), column_index), length(complete_index), 1);
            distance = power((temp - complete(:, column_index)), 2);
            distance = distance * (Mi_dataset(miss_atr_index(m), column_index))';
            distance = sqrt(distance);
            %[~, distance_index] = sort(distance, 'descend'); % 原始降序
            [~, distance_index] = sort(distance); % 修改为升序
            make_K = sum(distance) * K_proportion;
            sum_distance = 0;
            K = 1;
            while sum_distance < make_K && K <= length(distance)
                sum_distance = sum_distance + distance(distance_index(K));
                K = K + 1;
            end
    
            neighbor_values = complete(distance_index(1:K), miss_atr_index(m));
            
            % 修改距离以确保没有零值
            adjusted_distance = distance(distance_index(1:K)) + 1e-10;
            distance_p = 1 ./ adjusted_distance;
            distance_p = distance_p / sum(distance_p); 
        
            % 离散连续值处理
            % num_bins = K; % 原始注释
            num_bins = ceil(sqrt(row));
            bin_edges = linspace(min(complete(:, miss_atr_index(m))), max(complete(:, miss_atr_index(m))), num_bins + 1);
            binned_neighbor_values = discretize(neighbor_values, bin_edges);
        
            % 计算 P(F = f_target)
            value_counts = histcounts(complete(:, miss_atr_index(m)), bin_edges);
            value_probs = value_counts / sum(value_counts);
        
            % 构建加权矩阵 CW
            weight_matrix = zeros(K, K);
            for ka = 1:K
                bin_index = binned_neighbor_values(ka);
                if ~isnan(bin_index)
                    weight_matrix(ka, ka) = distance_p(ka) * value_probs(bin_index);
                end
            end
            for ka = 1:K
                weight_matrix(ka, ka) = RBF_function(weight_matrix(ka, ka), 0, SIGMA_DE2);
            end
    
            % 执行加权线性回归以估计缺失值
            pearson_A = complete(distance_index(1:K), :);
            x = pinv((pearson_A(:, column_index)' * weight_matrix)) * (no_complete(rate_rank(i), column_index)');
            predict_value = x' * weight_matrix * pearson_A(:, miss_atr_index(m));
            no_complete(rate_rank(i), miss_atr_index(m)) = predict_value;
        end
    end
    
