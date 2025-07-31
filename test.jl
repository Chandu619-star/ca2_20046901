using JLD2
using Flux
using OneHotArrays
using CSV
using DataFrames
using Statistics

# Function to calculate balanced accuracy as required by CA2
function bal_acc(model, test_x, test_y; **kwargs)
    # Ensure test_x is transposed correctly (features x samples)
    if size(test_x, 1) != size(model[1].weight, 2)
        test_x = test_x'
    end
    
    predictions = model(test_x)
    pred_classes = Flux.onecold(predictions, [0, 1]) .- 1
    true_classes = Flux.onecold(test_y, [0, 1]) .- 1
    
    # Calculate balanced accuracy
    tp = sum((pred_classes .== 1) .& (true_classes .== 1))
    tn = sum((pred_classes .== 0) .& (true_classes .== 0))
    fp = sum((pred_classes .== 1) .& (true_classes .== 0))
    fn = sum((pred_classes .== 0) .& (true_classes .== 1))
    
    # Handle edge cases
    sensitivity = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0.0
    
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return balanced_accuracy
end

# Load the trained model from JLD2 file
JLD2.@load "ca2_20046901_model.jld2" trained_params trained_st
println("Model loaded successfully.")

# Here you or your lecturer can provide test_x and test_y for evaluation
# Example usage (to be replaced with actual test data by the lecturer):
# test_x = ... 
# test_y = ... 
# accuracy = bal_acc(trained_st, test_x, test_y)
# println("Balanced Accuracy on test set: ", accuracy)

# Function to preprocess test data (same as training)
function preprocess_test_data(test_data, X_mean, X_std)
    # Remove unwanted columns as specified (fnlwgt and education)
    select!(test_data, Not([:fnlwgt, :education]))
    
    # Handle categorical variables - convert to numeric
    function encode_categorical!(df)
        for col in names(df)
            col_data = df[!, col]
            if eltype(col_data) <: AbstractString || any(x -> isa(x, AbstractString), col_data)
                # Handle missing values by replacing with "Unknown"
                col_data_clean = [ismissing(val) || val == "?" ? "Unknown" : string(val) for val in col_data]
                unique_vals = unique(col_data_clean)
                mapping = Dict(val => Float32(i) for (i, val) in enumerate(unique_vals))
                df[!, col] = [mapping[val] for val in col_data_clean]
            else
                # Convert numeric columns to Float32
                df[!, col] = Float32.(coalesce.(col_data, 0))
            end
        end
    end
    
    # Encode categorical variables
    encode_categorical!(test_data)
    
    # Extract features
    X_test = Matrix(select(test_data, Not(:label)))
    y_test = Int.(test_data.label .== maximum(test_data.label))
    
    # Normalize using training statistics
    X_test = (X_test .- X_mean) ./ X_std
    
    return Float32.(X_test), y_test
end

# Load the trained model as specified in CA2 requirements
JLD2.@load "ca2_20046901_model.jld2" trained_params trained_st

println("Model loaded successfully.")
println("Ready for testing on hold out dataset.")
println("")
println("Usage example:")
println("# test_x and test_y should be provided by lecturer")
println("# bal_acc_result = bal_acc(trained_st, test_x, test_y)")
