using CSV
using DataFrames
using Flux
using Statistics
using JLD2
using OneHotArrays
using Random

Random.seed!(42)

# Load data
data = CSV.File("train.csv") |> DataFrame
select!(data, Not([:fnlwgt, :education]))

# Advanced preprocessing with multiple encoding strategies
target_col = string.(data.label)
y_numeric = map(x -> contains(x, ">50K") ? 1.0 : 0.0, target_col)

# Create categorical mappings for consistency
categorical_mappings = Dict()

for col in names(data)
    if col == "label"
        continue
    end
    
    col_data = data[!, col]
    if eltype(col_data) <: AbstractString || any(x -> isa(x, AbstractString), col_data)
        # Advanced categorical encoding
        str_data = string.(coalesce.(col_data, "Unknown"))
        unique_vals = unique(str_data)
        
        # Target encoding
        encoding_map = Dict()
        for val in unique_vals
            mask = str_data .== val
            if sum(mask) > 0
                # Use smoothed target encoding
                target_mean = mean(y_numeric[mask])
                global_mean = mean(y_numeric)
                count = sum(mask)
                # Smoothing factor
                smooth_factor = 1 / (1 + exp(-(count - 10) / 5))
                encoding_map[val] = smooth_factor * target_mean + (1 - smooth_factor) * global_mean
            else
                encoding_map[val] = mean(y_numeric)
            end
        end
        
        categorical_mappings[col] = encoding_map
        data[!, col] = [Float32(encoding_map[val]) for val in str_data]
    else
        # Advanced numeric preprocessing
        col_clean = Float32.(coalesce.(col_data, median(skipmissing(col_data))))
        # Robust scaling
        q25, q75 = quantile(col_clean, [0.25, 0.75])
        iqr = q75 - q25 + 1e-8
        median_val = median(col_clean)
        data[!, col] = (col_clean .- median_val) ./ iqr
    end
end

# Comprehensive feature engineering
data.age_squared = data.age .^ 2
data.edu_age_interaction = data.age .* data.education_num  
data.capital_total = data.capital_gain .+ data.capital_loss
data.capital_ratio = (data.capital_gain .+ 1) ./ (data.capital_loss .+ 1)
data.hours_category = Float32.(data.hour_per_week .> 40)
data.high_education = Float32.(data.education_num .> 12)
data.prime_age = Float32.((data.age .> 25) .& (data.age .< 65))

# Prepare data
X = Matrix(select(data, Not(:label)))
y = Int.(y_numeric)

println("Original class distribution: ", sum(y), "/", length(y))

# Use original unbalanced data for better generalization
n_samples = size(X, 1)
n_train = Int(floor(0.8 * n_samples))

# Stratified split
pos_idx = findall(y .== 1)
neg_idx = findall(y .== 0)

n_pos_train = Int(floor(0.8 * length(pos_idx)))
n_neg_train = Int(floor(0.8 * length(neg_idx)))

train_pos = pos_idx[1:n_pos_train]
train_neg = neg_idx[1:n_neg_train]
test_pos = pos_idx[n_pos_train+1:end]
test_neg = neg_idx[n_neg_train+1:end]

train_idx = shuffle([train_pos; train_neg])
test_idx = shuffle([test_pos; test_neg])

X_train = Float32.(X[train_idx, :])
X_test = Float32.(X[test_idx, :])
y_train = y[train_idx]
y_test = y[test_idx]

println("Training samples: ", length(train_idx))
println("Test samples: ", length(test_idx))
println("Features: ", size(X_train, 2))
println("Train class distribution: ", sum(y_train), "/", length(y_train))

# Create optimized model architecture
n_features = size(X_train, 2)
model = Chain(
    Dense(n_features, 64, swish),
    Dropout(0.4),
    Dense(64, 32, swish), 
    Dropout(0.3),
    Dense(32, 16, swish),
    Dropout(0.2),
    Dense(16, 2)
)

total_params = sum(length, Flux.trainable(model))
println("Total parameters: ", total_params)

if total_params > 1000
    # Reduce model size to fit constraint
    model = Chain(
        Dense(n_features, 28, swish),
        Dropout(0.3),
        Dense(28, 14, swish),
        Dropout(0.2),
        Dense(14, 2)
    )
    total_params = sum(length, Flux.trainable(model))
    println("Adjusted parameters: ", total_params)
end

# Advanced training strategy
opt = Flux.setup(AdamW(0.001, (0.9, 0.999), 0.001), model)

best_acc = 0.0
patience = 0
max_patience = 20

for epoch in 1:100
    # Training with class weighting
    perm = randperm(length(y_train))
    
    for i in 1:128:length(y_train)
        end_i = min(i+127, length(y_train))
        batch_idx = perm[i:end_i]
        
        X_batch = X_train[batch_idx, :]'
        y_batch = onehotbatch(y_train[batch_idx], [0, 1])
        
        # Apply class weights for imbalanced data
        class_weights = Float32.([0.7, 1.3])  # Weight positive class more
        weighted_y_batch = y_batch .* reshape(class_weights, :, 1)
        
        loss_val, grads = Flux.withgradient(model) do m
            pred = m(X_batch)
            Flux.logitcrossentropy(pred, weighted_y_batch)
        end
        
        Flux.update!(opt, model, grads[1])
    end
    
    # Evaluate on test set
    if epoch % 5 == 0
        test_pred = model(X_test')
        pred_classes = Flux.onecold(test_pred, [0, 1]) .- 1
        
        tp = sum((pred_classes .== 1) .& (y_test .== 1))
        tn = sum((pred_classes .== 0) .& (y_test .== 0))
        fp = sum((pred_classes .== 1) .& (y_test .== 0))
        fn = sum((pred_classes .== 0) .& (y_test .== 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        bal_acc = (recall + specificity) / 2
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        println("Epoch $epoch:")
        println("  Balanced Accuracy: $bal_acc")
        println("  Accuracy: $accuracy")
        println("  F1 Score: $f1")
        println("  Precision: $precision")
        println("  Recall: $recall")
        
        if bal_acc > best_acc
            global best_acc = bal_acc
            patience = 0
            # Save best model
            global best_model_params = deepcopy(Flux.state(model))
        else
            global patience += 1
        end
        
        if patience >= max_patience
            println("Early stopping at epoch $epoch")
            break
        end
    end
end

# Restore best model
if @isdefined(best_model_params)
    Flux.loadmodel!(model, best_model_params)
end

# Final balanced accuracy function
function bal_acc(model, test_x, test_y; kwargs...)
    if size(test_x, 1) != size(model[1].weight, 2)
        test_x = test_x'
    end
    
    predictions = model(test_x)
    pred_classes = Flux.onecold(predictions, [0, 1]) .- 1
    
    if isa(test_y, Vector)
        true_classes = test_y
    else
        true_classes = Flux.onecold(test_y, [0, 1]) .- 1
    end
    
    tp = sum((pred_classes .== 1) .& (true_classes .== 1))
    tn = sum((pred_classes .== 0) .& (true_classes .== 0))
    fp = sum((pred_classes .== 1) .& (true_classes .== 0))
    fn = sum((pred_classes .== 0) .& (true_classes .== 1))
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return (sensitivity + specificity) / 2
end

final_accuracy = bal_acc(model, X_test', y_test)

println("\n" * "="^50)
println("FINAL RESULTS")
println("="^50)
println("Final Balanced Accuracy: $final_accuracy")
println("Best Training Accuracy: $best_acc")
println("Model Parameters: $total_params/1000")

# Save model
trained_params = Flux.state(model)
trained_st = model

JLD2.@save "ca2_20046901_model.jld2" trained_params trained_st categorical_mappings

println("Model saved successfully with preprocessing mappings!")
