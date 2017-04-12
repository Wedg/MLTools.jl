function polynomial_features(data_in::Array; dim::Int=2,
                             subset::Vector=collect(1:size(data_in, dim)), degree::Int=2,
                             include_bias::Bool=true, interaction_only::Bool=false)

    # Data type of the elements of the input array
    T = eltype(data_in)

    # If features are in columns and examples are in rows
    if dim == 2

        # Size of original input matrix
        num_obs, num_feats_in = size(data_in)

        # Create matrix of the chosen subset of features that will generate the new polynomial features
        data_sub = data_in[:, subset]

        # How many original features are chosen to generate the polynomial features
        num_feats_sub = length(subset)

        # If we're only generating the interaction terms (i.e. only single power cross terms)
        if interaction_only

            # Number of columns needed for the new features
            num_feats_poly = 0
            for deg = 2:degree
                num_feats_poly += binomial(num_feats_sub, deg)
            end

            # Initialise design matrix with:
            # column for bias if wanted i.e. if include_bias = true
            # columns for original features i.e. data_in
            # columns for the polynomial features that will be generated (initialised to zero)
            if include_bias
                data_out = [ones(T, num_obs) data_in zeros(T, num_obs, num_feats_poly)]
            else
                data_out = [data_in zeros(T, num_obs, num_feats_poly)]
            end

            # Generate polynomial features and add to data_out
            ind = (include_bias ? 1 : 0) + num_feats_in + 1        # column number to start placing the new features
            for deg = 2:degree                                     # generate features for each power - 2:degree
                for c in combinations(collect(1:num_feats_sub), deg)      # create combinations iterator
                    p = zeros(Int, num_feats_sub)                  # zeros vector of length = number of features in the subset
                    p[c] = 1                                       # change this combination's elements to 1
                    data_out[:, ind] = prod(data_sub .^ p', 2)     # use the power vector to create the new feature
                    ind += 1                                       # move one column to the right to place next new feature
                end
            end

        # If we're generating all the terms
        else

            # Number of columns needed for the new features
            num_feats_poly = 0
            for deg = 2:degree
                num_feats_poly += binomial(num_feats_sub + deg - 1, num_feats_sub - 1)
            end

            # Initialise design matrix with:
            # column for bias if wanted i.e. if include_bias = true
            # columns for original features i.e. data_in
            # columns for the polynomial features that will be generated (initialised to zero)
            if include_bias
                data_out = [ones(T, num_obs) data_in zeros(T, num_obs, num_feats_poly)]
            else
                data_out = [data_in zeros(T, num_obs, num_feats_poly)]
            end

            # Generate polynomial features and add to data_out
            ind = (include_bias ? 1 : 0) + num_feats_in + 1                                 # column number to start placing the new features
            for deg = 2:degree                                                              # generate features for each power - 2:degree
                tmp = [(num_feats_sub + deg); zeros(Int, num_feats_sub)]                    # temp vector used to convert format of combination
                for c in combinations(collect(1:(num_feats_sub + deg - 1)), num_feats_sub - 1)     # create combinations iterator
                    tmp[num_feats_sub:-1:2] = c                                             # place combination into tmp vector
                    p = -diff(tmp)-1                                                        # vector of powers
                    data_out[:, ind] = prod(data_sub .^ p', 2)                              # use the power vector to create the new feature
                    ind += 1                                                                # move one column to the right to place next new feature
                end
            end

        end

    # If features are in rows and examples are in cols
    elseif dim == 1

        # Size of original input matrix
        num_feats_in, num_obs = size(data_in)

        # Create matrix of the chosen subset of features that will generate the new polynomial features
        data_sub = data_in[subset, :]

        # How many original features are chosen to generate the polynomial features
        num_feats_sub = length(subset)

        # If we're only generating the interaction terms (i.e. only single power cross terms)
        if interaction_only

            # Number of rows needed for the new features
            num_feats_poly = 0
            for deg = 2:degree
                num_feats_poly += binomial(num_feats_sub, deg)
            end

            # Initialise design matrix with:
            # row for bias if wanted i.e. if include_bias = true
            # rows for original features i.e. data_in
            # rows for the polynomial features that will be generated (initialised to zero)
            if include_bias
                data_out = [ones(T, 1, num_obs) ; data_in ; zeros(T, num_feats_poly, num_obs)]
            else
                data_out = [data_in ; zeros(T, num_feats_poly, num_obs)]
            end

            # Generate polynomial features and add to data_out
            ind = (include_bias ? 1 : 0) + num_feats_in + 1       # row number to start placing the new features
            for deg = 2:degree                                    # generate features for each power - 2:degree
                for c in combinations(collect(1:num_feats_sub), deg)     # create combinations iterator
                    p = zeros(Int, num_feats_sub)                 # zeros vector of length = number of features in the subset
                    p[c] = 1                                      # change this combination's elements to 1
                    data_out[ind, :] = prod(data_sub .^ p, 1)     # use the power vector to create the new feature
                    ind += 1                                      # move one column to the right to place next new feature
                end
            end

        # If we're generating all the terms
        else

            # Number of columns needed for the new features
            num_feats_poly = 0
            for deg = 2:degree
                num_feats_poly += binomial(num_feats_sub + deg - 1, num_feats_sub - 1)
            end

            # Initialise design matrix with:
            # row for bias if wanted i.e. if include_bias = true
            # rows for original features i.e. data_in
            # rows for the polynomial features that will be generated (initialised to zero)
            if include_bias
                data_out = [ones(T, 1, num_obs) ; data_in ; zeros(T, num_feats_poly, num_obs)]
            else
                data_out = [data_in ; zeros(T, num_feats_poly, num_obs)]
            end

            # Generate polynomial features and add to data_out
            ind = (include_bias ? 1 : 0) + num_feats_in + 1                                 # row number to start placing the new features
            for deg = 2:degree                                                              # generate features for each power - 2:degree
                tmp = [(num_feats_sub + deg); zeros(Int, num_feats_sub)]                    # temp vector used to convert format of combination
                for c in combinations(collect(1:(num_feats_sub + deg - 1)), num_feats_sub - 1)     # create combinations iterator
                    tmp[num_feats_sub:-1:2] = c                                             # place combination into tmp vector
                    p = -diff(tmp)-1                                                        # vector of powers
                    data_out[ind, :] = prod(data_sub .^ p, 1)                               # use the power vector to create the new feature
                    ind += 1                                                                # move one column to the right to place next new feature
                end
            end

        end

    # If dim entered is not in range i.e. must be 1 or 2
    else throw(DimensionMismatch("Dimension must be 1 (features are rows) or 2, the default, (features are cols)."))

    end

    return data_out
end
