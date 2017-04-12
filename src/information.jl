# =============== String Conversion ===============

# Convert vector of strings to vector of integers to enable histogram for string vector
function stringvec2intvec{Ts<:AbstractString}(s::Array{Ts,1})
    u = unique(s)
    x = zeros(Int, length(s))
    for i = 1:length(u)
        x[find(s .== u[i])] = i
    end
    return x
end

# =============== Bin Calcs ===============

# Calculate number of bins in histogram for vector of floating point numbers

# Freedman-Diaconis rule
function nBins_fdr{Tf<:AbstractFloat}(x::Array{Tf,1})
    n = length(x)
    iqr = quantile(x, 0.75) - quantile(x, 0.25)
    maxmin = maximum(x) - minimum(x)
    fdr = 2 * iqr * n^(-1/3)
    nbins = ceil(Int, maxmin/fdr)
end

# Scott's Rule
function nBins_sr{Tf<:AbstractFloat}(x::Array{Tf,1})
    n = length(x)
    iqr = quantile(x, 0.75) - quantile(x, 0.25)
    maxmin = maximum(x) - minimum(x)
    sr = 3.5*std(x)*(length(x))^(-1/3)
    nbins = ceil(Int, maxmin/sr)
end

# Square Root method
function nBins_sqrt{Tf<:AbstractFloat}(x::Array{Tf,1})
    n = length(x)
    nbins = ceil(Int, sqrt(n))
end

# =============== Entropy ===============

# Entropy fuctions for vectors of integers, real numbers and strings

function entropy{Ti<:Integer}(x::Array{Ti,1})
    n = length(x)                            # n = total number of values
    nBins = length(unique(x))                # nBins = number of unique values
    c = hist(x, nBins)[2]                    # c = vector showing count of values of each outcome (i.e. in each bucket)
    p = c ./ n                               # p = probability of each outcome (i.e. bucket)
    #p = P[P .!= 0]                          # remove zeros [not necessary]
    H = -sum(p .* log2(p))                   # H = entropy of x
end

function entropy{Tf<:AbstractFloat}(x::Array{Tf,1})
    n = length(x)                            # n = total number of values
    nBins = nBins_fdr(x)                     # nBins = number of bins using Freedman-Diaconis rule
    c = hist(x, nBins)[2]                    # c = vector showing count of values in each bucket
    P = c ./ n                               # P = probability of each bucket
    p = P[P .!= 0]                           # remove zeros
    H = -sum(p .* log2(p))                   # H = entropy of x
end

function entropy{Ts<:AbstractString}(s::Array{Ts,1})
    x = stringvec2intvec(s)                  # Change to vector of integers
    entropy(x)                               # Calculate entropy using Integer method
end

# =============== Joint Entropy ===============

# Joint Entropy functions for vectors of integers, real numbers and strings

function jointEntropy{Ti<:Integer}(x::Array{Ti,1}, y::Array{Ti,1})
    n = length(x)                             # n = total number of values (same length for both x and y)
    nBinsx = length(unique(x))                # nBinsx = number of unique values of vector x
    nBinsy = length(unique(y))                # nBinsy = number of unique values of vector y
    C = hist2d([x y], nBinsx, nBinsy)[3]      # C = matrix showing count of values in each joint outcome (i.e. in each bucket)
    P = C ./ n                                # P = probability of each joint outcome (i.e. bucket)
    p = P[P .!= 0]                            # remove zeros
    H = -sum(p .* log2(p))                    # H = joint entropy of x and y
end

function jointEntropy{Tf<:AbstractFloat, Ti<:Integer}(x::Array{Tf,1}, y::Array{Ti,1})
    n = length(x)                             # n = total number of values (same length for both x and y)
    nBinsx = nBins_fdr(x)                     # nBinsx = number of bins of vector x using Freedman-Diaconis rule
    nBinsy = length(unique(y))                # nBinsy = number of unique values of vector y
    C = hist2d([x y], nBinsx, nBinsy)[3]      # C = matrix showing count of values in each bucket
    P = C ./ n                                # P = probability of each bucket
    p = P[P .!= 0]                            # remove zeros
    H = -sum(p .* log2(p))                    # H = joint entropy of x and y
end

function jointEntropy{Ti<:Integer, Tf<:AbstractFloat}(x::Array{Ti,1}, y::Array{Tf,1})
    n = length(x)                             # n = total number of values (same length for both x and y)
    nBinsx = length(unique(x))                # nBinsx = number of unique values of vector x
    nBinsy = nBins_fdr(y)                     # nBinsy = number of bins of vector y using Freedman-Diaconis rule
    C = hist2d([x y], nBinsx, nBinsy)[3]      # C = matrix showing count of values in each bucket
    P = C ./ n                                # P = probability of each bucket
    p = P[P .!= 0]                            # remove zeros
    H = -sum(p .* log2(p))                    # H = joint entropy of x and y
end

function jointEntropy{Tf<:AbstractFloat}(x::Array{Tf,1}, y::Array{Tf,1})
    n = length(x)                             # n = total number of values (same length for both x and y)
    nBinsx = nBins_fdr(x)                     # nBinsx = number of bins of vector x using Freedman-Diaconis rule
    nBinsy = nBins_fdr(y)                     # nBinsy = number of bins of vector y using Freedman-Diaconis rule
    C = hist2d([x y], nBinsx, nBinsy)[3]      # C = matrix showing count of values in each bucket
    P = C ./ n                                # P = probability of each bucket
    p = P[P .!= 0]                            # remove zeros
    H = -sum(p .* log2(p))                    # H = joint entropy of x and y
end

function jointEntropy{Ts<:AbstractString}(s1::Array{Ts,1}, s2::Array{Ts,1})
    x = stringvec2intvec(s1)
    y = stringvec2intvec(s2)
    jointEntropy(x, y)
end

function jointEntropy{Ts<:AbstractString, Tr<:Real}(s::Array{Ts,1}, y::Array{Tr,1})
    x = stringvec2intvec(s)
    jointEntropy(x, y)
end

function jointEntropy{Tr<:Real, Ts<:AbstractString}(x::Array{Tr,1}, s::Array{Ts,1})
    y = stringvec2intvec(s)
    jointEntropy(x, y)
end

# =============== Mutual Information ===============

# Mutual Information function for vectors of integers, real numbers and strings
function mutualInformation(x, y)
    entropy(x) + entropy(y) - jointEntropy(x, y)
end

# Mutual Information function for vectors within dataframes
function mutualInformation(MIdf::DataFrame, dx::Symbol, dy::Symbol)
    df = MIdf[:, [dx, dy]]
    complete_cases!(df)
    x = convert(Array, df[dx])
    y = convert(Array, df[dy])
    mutualInformation(x, y)
    #entropy(x) + entropy(y) - jointEntropy(x, y)
end

# Mutual information table from dataframe
function mutualInformation(MIdf::DataFrame, target::Symbol)
    variables = names(MIdf)
    targetindex = findfirst(variables, target)
    splice!(variables, targetindex)
    MIvec = Real[]
    for var in variables
        df = MIdf[:, [var, target]]
        complete_cases!(df)
        x = convert(Array, df[var])
        y = convert(Array, df[target])
        push!(MIvec, mutualInformation(x, y))
    end
    df = DataFrame(Name=variables)
    insert!(df, 2, MIvec, :MI)
    sort(df, cols=:MI, rev=true)
end
