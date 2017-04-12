# Normalises array with options to choose:
# 1) dimension along which to normalise: dim=1 means features are rows, dim=2 means cols
# 2) subset of features to be normalised
# returns new array X_out and μ and σ
function normalise(X::Array; dim::Int=2, subset::Vector=collect(1:size(X, dim)))

    # Normalise each column - default
    if dim == 2
        data = X[:, subset]
        μ = mean(data, 1)
        σ = std(data, 1)
        X_out = copy(X)
        for i = 1:length(subset)
            X_out[:, subset[i]] = (X_out[:, subset[i]] .- μ[i]) ./ σ[i]
        end
        return X_out, μ, σ

    # Normalise each row
    elseif dim == 1
        data = X[subset, :]
        μ = mean(data, 2)
        σ = std(data, 2)
        X_out = copy(X)
        for i = 1:length(subset)
            X_out[subset[i], :] = (X_out[subset[i], :] .- μ[i]) ./ σ[i]
        end
        return X_out, μ, σ

    # If dim entered is not in range i.e. must be 1 or 2
    else throw(DimensionMismatch("Dimension must be 1 (features are rows) or 2, the default, (features are cols)."))

    end
end

# Normalises array with options to choose:
# 1) dimension along which to normalise: dim=1 means features are rows, dim=2 means cols
# 2) subset of features to be normalised
# modifies X and returns μ and σ
function normalise!(X::Array; dim::Int=2, subset::Vector=collect(1:size(X, dim)))

    # Normalise each column - default
    if dim == 2
        data = X[:, subset]
        μ = mean(data, 1)
        σ = std(data, 1)
        for i = 1:length(subset)
            X[:, subset[i]] = (X[:, subset[i]] .- μ[i]) ./ σ[i]
        end
        return μ, σ

    # Normalise each row
    elseif dim == 1
        data = X[subset, :]
        μ = mean(data, 2)
        σ = std(data, 2)
        for i = 1:length(subset)
            X[subset[i], :] = (X[subset[i], :] .- μ[i]) ./ σ[i]
        end
        return μ, σ

    # If dim entered is not in range i.e. must be 1 or 2
    else throw(DimensionMismatch("Dimension must be 1 (features are rows) or 2, the default, (features are cols)."))

    end
end

# Normalises array with given μ (mean) and σ (standard deviation)
# Same options as above for dimension and subset of features
function normalise(X::Array, μ::Array, σ::Array; dim::Int=2, subset::Vector=collect(1:size(X, dim)))

    # Check lengths
    length(μ) != length(σ) && throw("μ and σ must be same length")
    length(μ) != length(subset) && throw("μ and subset must be same length")

    # Normalise each column - default
    if dim == 2
        X_out = copy(X)
        for i = 1:length(subset)
            X_out[:, subset[i]] = (X_out[:, subset[i]] .- μ[i]) ./ σ[i]
        end
        return X_out

    # Normalise each row
    elseif dim == 1
        X_out = copy(X)
        for i = 1:length(subset)
            X_out[subset[i], :] = (X_out[subset[i], :] .- μ[i]) ./ σ[i]
        end
        return X_out

    # If dim entered is not in range i.e. must be 1 or 2
    else throw(DimensionMismatch("Dimension must be 1 (features are rows) or 2, the default, (features are cols)."))

    end
end

# Normalises array with given μ (mean) and σ (standard deviation)
# Same options as above for dimension and subset of features
function normalise!(X::Array, μ::Array, σ::Array; dim::Int=2, subset::Vector=collect(1:size(X, dim)))

    # Check lengths
    length(μ) != length(σ) && throw("μ and σ must be same length")
    length(μ) != length(subset) && throw("μ and subset must be same length")

    # Normalise each column - default
    if dim == 2
        for i = 1:length(subset)
            X[:, subset[i]] = (X[:, subset[i]] .- μ[i]) ./ σ[i]
        end

    # Normalise each row
    elseif dim == 1
        for i = 1:length(subset)
            X[subset[i], :] = (X[subset[i], :] .- μ[i]) ./ σ[i]
        end

    # If dim entered is not in range i.e. must be 1 or 2
    else throw(DimensionMismatch("Dimension must be 1 (features are rows) or 2, the default, (features are cols)."))

    end
end
