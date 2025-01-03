function imputed2 = SCMLLSImpute_continuity(missing, sigma, K_proportion)
% SCMLLSImpute_continuity Impute missing continuous data using Structural Causal Model with LLSimpute.
%
%   imputed2 = SCMLLSImpute_continuity(missing)
%   imputed2 = SCMLLSImpute_continuity(missing, sigma)
%   imputed2 = SCMLLSImpute_continuity(missing, sigma, K_proportion)
%
%   Inputs:
%       missing        - [p x q] dataset with missing values denoted by 9
%       sigma          - [optional] RBF kernel parameter (default: 0.09)
%       K_proportion   - [optional] Proportion for dynamic neighbor selection (default: 0.06)
%
%   Output:
%       imputed2       - [p x q] imputed dataset

    % Set default parameters if not provided
    if nargin < 3
        K_proportion = 0.06;
        if nargin < 2
            sigma = 0.09;
        end
    end

    complete = [];
    no_complete = [];
    no_complete_index = [];
    complete_index = [];
    [row, col] = size(missing);
    Mi_dataset = zeros(col, col);
    gauss_sum = zeros(col, 1);
    gauss_dataset = zeros(col, col);
    logic_dataset = zeros(col, col);   
    
    % Identify incomplete instances (rows with at least one missing value denoted by 9)
    for i = 1:row
        if isempty(find(missing(i, :) == 9, 1))
            continue;
        else
            no_complete_index = [no_complete_index i];
        end
    end
    
    no_complete = missing(no_complete_index, :);
    complete_index = setdiff(1:row, no_complete_index);
    complete = missing(complete_index, :);
    
    % Calculate the mutual information (MI) correlation matrix
    for k = 1:col
        for j = 1:col
            if k ~= j
                Mi_dataset(k, j) = MI(missing(:, j), missing(:, k));
            else
                Mi_dataset(k, j) = 1;  
            end
        end
    end
    
    % Apply the RBF kernel to the MI matrix to create the weight matrix
    for k = 1:col
        gauss_sum(k) = 0;
        for j = 1:col
            gauss_dataset(k, j) = RBF_function(1, Mi_dataset(k, j), sigma);
            gauss_sum(k) = gauss_sum(k) + gauss_dataset(k, j);
        end
    end
    
    % Normalize the Gaussian sums and cap at the number of features
    gauss_sum = round((gauss_sum / max(gauss_sum)) * col + col);
    gauss_sum(gauss_sum > col) = col;
    Mi_dataset = zeros(col, col);
    
    % Select top K neighbors based on the normalized Gaussian weights
    for k = 1:col    
        real_index = MAX_K_NUMBER(gauss_dataset(k, :), gauss_sum(k));
        Mi_dataset(k, real_index) = gauss_dataset(k, real_index);
    end
        
    imputed1 = Mi_dataset;
    SIGMA_DE2 = 2;
    missing_rate = [];
    
    % Calculate missing rates (currently unused but can be utilized for analysis)
    for no_com = 1:length(no_complete_index)
        nan_index = find(no_complete(no_com, :) == 9);
        nan_num = length(nan_index);
        nan_rate = nan_num / col;
        % missing_rate = [missing_rate nan_rate];
    end
    [~, rate_rank] = sort(missing_rate);
    
    % Missing data imputation
    for i = 1:length(rate_rank)
        if ~any(~(no_complete(rate_rank(i), :) == 9))
            continue;
        end
        miss_row_index = rate_rank(i);
        miss_atr_index = find(no_complete(rate_rank(i), :) == 9);
        miss_atr_num = length(miss_atr_index);
        column_index1 = find(no_complete(rate_rank(i), :) ~= 9);
        
        for m = 1:miss_atr_num
            % Find relevant columns for the current missing attribute
            column_index = intersect(column_index1, find(Mi_dataset(miss_atr_index(m), :) ~= 0));
            temp = repmat(no_complete(rate_rank(i), column_index), length(complete_index), 1);
            distance = power((temp - complete(:, column_index)), 2);
            distance = distance * (Mi_dataset(miss_atr_index(m), column_index))';
            distance = sqrt(distance);
            
            % Sort distances in ascending order
            [~, distance_index] = sort(distance);
            
            % Determine the number of neighbors based on K_proportion
            make_K = sum(distance) * K_proportion;
            sum_distance = 0;
            K = 1;
            while sum_distance < make_K && K <= length(distance)
                sum_distance = sum_distance + distance(distance_index(K));
                K = K + 1;
            end
        
            % Extract neighbor values for the current missing attribute
            neighbor_values = complete(distance_index(1:K), miss_atr_index(m));
            
            % Adjust distances to avoid division by zero
            adjusted_distance = distance(distance_index(1:K)) + 1e-10;
            distance_p = 1 ./ adjusted_distance;
            distance_p = distance_p / sum(distance_p); 
    
            % Discretize neighbor values into bins for stability
            num_bins = ceil(sqrt(row));
            bin_edges = linspace(min(complete(:, miss_atr_index(m))), max(complete(:, miss_atr_index(m))), num_bins + 1);
            binned_neighbor_values = discretize(neighbor_values, bin_edges);
        
            % Compute P(F=f_target)
            value_counts = histcounts(complete(:, miss_atr_index(m)), bin_edges);
            value_probs = value_counts / sum(value_counts);
        
            % Construct the weighting matrix CW
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
    
            % Perform weighted linear regression to estimate the missing value
            pearson_A = complete(distance_index(1:K), :);
            x = pinv((pearson_A(:, column_index)' * weight_matrix)) * (no_complete(rate_rank(i), column_index)');
            predict_value = x' * weight_matrix * pearson_A(:, miss_atr_index(m));
            no_complete(rate_rank(i), miss_atr_index(m)) = predict_value;
        end
    end
    
    % Update the original dataset with imputed values
    missing(no_complete_index, :) = no_complete;
    imputed2 = missing;
end

% Auxiliary Functions (Assumed to be defined elsewhere)

function mi = MI(X, Y)
    % MI Mutual Information between X and Y
    % Implement mutual information calculation
    % Placeholder implementation; replace with actual MI computation
    mi = mutualinfo(X, Y); % Assume mutualinfo is a predefined function
end

function val = RBF_function(x, y, sigma)
    % RBF_function Radial Basis Function
    val = exp(-((x - y)^2) / (2 * sigma^2));
end

function topK_indices = MAX_K_NUMBER(array, K)
    % MAX_K_NUMBER Select indices of the top K largest values in 'array'
    [~, sorted_indices] = sort(array, 'descend');
    K = min(K, length(sorted_indices)); % Ensure K does not exceed array length
    topK_indices = sorted_indices(1:K);
end
