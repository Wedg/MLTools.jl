function kFoldClassPropnPrep(X::Array, y::Vector, numFolds::Int)

  classes = sort(unique(y))                                     # Vector containing names of class outcomes
  numClasses = length(classes)                                  # Number of output classes
  numPerClass = Int[sum(y .== k) for k in classes]              # Vector containing number of instances of each class

  # Check that there are at least 'numFolds' instances of each class
  for k = 1:numClasses
      numPerClass[k] < numFolds && display("ERROR! Label $k has fewer than $numFolds (number of folds) samples.")
  end

  # FoldSizes will be a matrix holding the number of vectors to place in each fold for each category.
  # The number of folds may not divide evenly into the number of vectors so we need to distribute the remainder.
  FoldSizes = zeros(Int, numClasses, numFolds)                  # Initiate empty FoldSizes matrix
  for k = 1:numClasses                                          # For each class ...
    num_k = numPerClass[k]                                      #  record the number of instances in that class
    for f = 1:numFolds                                          #  Then for each fold ...
      foldSize = int(ceil(num_k / (numFolds - f + 1)))          #   calculate the allocation from this class to this fold
      FoldSizes[k, f] = foldSize                                #   record that number in the FoldSizes matrix
      num_k -= foldSize                                         #   reduce the number to be allocated to the remaining folds
    end
  end

  # Check the fold sizes sum up correctly by comparing to the total number for each class
  any(sum(FoldSizes, 2) .!= numPerClass) && display("ERROR! The sum of fold sizes does not equal the number of vectors.")

  # Randomly sort the matrix X and vector y and then group by class for proportional sampling
  m, n = size(X)                                                # m = # of examples, n = # of features
  randVecs = [X y][randperm(m), :]                              # Randomly shuffle the input vectors and output classes
  groupVecs = sortrows(randVecs, by=x->x[n + 1])                # Group the vectors by class for proportional sampling
  X_sorted = groupVecs[:, 1:n]                                  # Retrieve shuffled and sorted X
  y_sorted = groupVecs[:, n + 1]                                # Retrieve shuffled and sorted y

  # Calculate arrays containing starting and ending indices of each class for each fold
  cvStart = cumsum(numPerClass) .- numPerClass .+ 1 .+ cumsum(FoldSizes, 2) .- FoldSizes
  cvEnd = cvStart .+ FoldSizes - 1

  # Return everything needed for the k-fold CV loop
  return X_sorted, y_sorted, classes, FoldSizes, cvStart, cvEnd
end

function kFoldClassPropnArrays(X_sorted::Array, y_sorted::Vector, classes::Vector,
                               cvStart::Array{Int,2}, cvEnd::Array{Int,2}, roundNumber::Int)

  numClasses = length(classes)                                  # Number of output classes
  numVectors = length(y_sorted)                                 # Number of vectors

  cv_inds = Int[]                                               # Ititialise empty vector to contain cv set indices
  for k = 1 : numClasses                                        # For each class ...
    append!(cv_inds, cvStart[k, roundNumber] : cvEnd[k, roundNumber])    #  fetch the cv set indices for this round
  end
  train_inds = setdiff(1:numVectors, cv_inds)                   # Vector of training set indices != cv set indices

  X_train = X_sorted[train_inds, :]                             # Matrix containing training set vectors
  y_train = y_sorted[train_inds]                                # Vector containing training set classes
  X_cv = X_sorted[cv_inds, :]                                   # Matrix containing cross validation set vectors
  y_cv = y_sorted[cv_inds]                                      # Vector containing cross validation set classes

  return X_train, y_train, X_cv, y_cv                           # Return training and cross validation arrays
end

function shufflePartition2(X::Array, y::Vector, p_cv::Float64=0.2)

  # Function
  # Inputs
  # Outputs

  num_cv = int(length(y) * p_cv)                                # Number in cross validation set
  cv_inds = shuffle([1:length(y)] .<= num_cv)                   # Vector of shuffled indices

  X_train = X[!cv_inds, :]                                      # Matrix containing training set vectors
  y_train = y[!cv_inds]                                         # Vector containing training set classes
  X_cv = X[cv_inds, :]                                          # Matrix containing cross validation set vectors
  y_cv = y[cv_inds]                                             # Vector containing cross validation set classes

  return X_train, y_train, X_cv, y_cv                           # Return training and cross validation arrays
end

function shufflePartition3(X::Array, y::Vector, p_cv::Float64=0.2, p_test::Float64=0.2)

  num_cv = int(length(y) * p_cv)                                # Number in cross validation set
  num_test = int(length(y) * p_test)                            # Number in test set
  num_train = int(length(y)) - num_cv - num_test                # Number in training set

  # Vector of shuffled indices
  inds = shuffle([ones(Int, num_train); ones(Int, num_cv)*2; ones(Int, num_test)*3])

  X_train = X[inds .== 1, :]                                    # Matrix containing training set vectors
  y_train = y[inds .== 1]                                       # Vector containing training set classes
  X_cv = X[inds .== 2, :]                                       # Matrix containing cross validation set vectors
  y_cv = y[inds .== 2]                                          # Vector containing cross validation set classes
  X_test = X[inds .== 3, :]                                     # Matrix containing test set vectors
  y_test = y[inds .== 3]                                        # Vector containing test set classes

  return X_train, y_train, X_cv, y_cv, X_test, y_test           # Return training, cv and test arrays
end

function propnPartition2(X::Array, y::Vector, p_cv::Float64=0.2)

  classes = sort(unique(y))                                     # Vector containing names of class outcomes
  numClasses = length(classes)                                  # Number of output classes
  numPerClass = Int[sum(y .== k) for k in classes]              # Vector containing number of instances of each class
  num_cv = int(length(y) * p_cv)                                # Number in cross validation set

  # Randomly sort the matrix X and vector y and then group by class for proportional sampling
  m, n = size(X)                                                # m = # of examples, n = # of features
  randVecs = [X y][randperm(m), :]                              # Randomly shuffle the input Matrix and output Vector
  groupVecs = sortrows(randVecs, by=x->x[n + 1])                # Group the vectors by class for proportional sampling
  X_sorted = groupVecs[:, 1:n]                                  # Retrieve shuffled and sorted X Matrix
  y_sorted = groupVecs[:, n + 1]                                # Retrieve shuffled and sorted y Vector

  cv_size = int(numPerClass * p_cv)                             # Vector of size of each class in the cv set
  cv_start = cumsum(numPerClass) .- numPerClass + 1             # Vector of start indices of each class
  cv_end = cv_start .+ cv_size - 1                              # Vector of end indices in cv set of each class

  cv_inds = Int[]                                               # Ititialise empty vector to contain cv set indices
  for k = 1 : numClasses                                        # For each class ...
    append!(cv_inds, cv_start[k] : cv_end[k])                   #  fetch the cv set indices for this round
  end
  train_inds = setdiff(1:m, cv_inds)                            # Vector of training set indices != cv set indices

  X_train = X_sorted[train_inds, :]                             # Matrix containing training set vectors
  y_train = y_sorted[train_inds]                                # Vector containing training set classes
  X_cv = X_sorted[cv_inds, :]                                   # Matrix containing cross validation set vectors
  y_cv = y_sorted[cv_inds]                                      # Vector containing cross validation set classes

  return X_train, y_train, X_cv, y_cv                           # Return training and cross validation arrays
end