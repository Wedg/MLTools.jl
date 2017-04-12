# Create confusion matrix for any number of classes
function confusion_matrix(y_cond::Vector{Int}, y_pred::Vector{Int})

    # Generate empty confusion matrix
    classes = sort(unique([y_cond; y_pred]))
    conf_matrix = zeros(Int64, length(classes), length(classes))

    # Add counts
    for i in 1:length(y_cond)
        # translate label to index
        cond_class_index = findfirst(classes, y_cond[i])
        pred_class_index = findfirst(classes, y_pred[i])
        # predicted class is the row, condition class is the column
        conf_matrix[pred_class_index, cond_class_index] += 1
  end

  return conf_matrix
end

# plot confusion matrix
function plot_confusion_matrix(y_cond::Vector{Int},
                               y_pred::Vector{Int},
                               ylabel::String = "Var1",
                               xlabel::String = "Var2",
                               classes::Vector{String} = string.(sort(unique([y_cond; y_pred]))))
    # Number of classes
    num = length(classes)

    # Create DataFrame with data from confusion matrix
    axes = vcat([[classes[i] classes[j]] for i in 1:num, j in 1:num]...)
    df = DataFrame([axes confusion_matrix(y_cond, y_pred)[:]])

    # Colormap
    cmap = vcat(RGB(1,1,1), sequential_palette(255, 99, c=0.88, s=0.6, b=0.75, w=0.3, d=0.25,
                            wcolor=RGB(1,1,1), dcolor=RGB(0,0,1), logscale=false))

    # Plot
    plot(df, x="x2", y="x1", color="x3", Geom.rectbin,
         Coord.cartesian(yflip=true, xflip=false, fixed=true),
         Guide.YLabel(ylabel), Guide.XLabel(xlabel), Guide.colorkey("Count"),
         Scale.color_continuous(colormap = p->cmap[round(Int, p*99+1)], minvalue=0))
end

# Print accuracy stats
# Optional to print short, medium or long version
function print_binary_accuracy(y_cond::Vector{Int}, y_pred::Vector{Int}; size="short")

    # Check only 2 classes
    length(unique([y_cond; y_pred])) > 2 && throw("function is only for 2 classes")

    # Create confusion matrix
    CM = confusion_matrix(y_cond, y_pred)

    # Calculate metrics

    # 1st level metrics
    Population = sum(CM)          # Total number in population

    TP = CM[1, 1]                 # "True Positive" / "Hit"
    TN = CM[2, 2]                 # "True Negative" / "Correct Reject"
    FP = CM[1, 2]                 # "False Positive" / "False Alarm" / Type I Error
    FN = CM[2, 1]                 # "False Negative" / "Miss" / Type II Error

    CondPos = TP + FN             # Sum of Condition = Positive
    CondNeg = FP + TN             # Sum of Condition = Negative
    PredPos = TP + FP             # Sum of Prediction = Positive
    PredNeg = FN + TN             # Sum of Prediction = Negative

    # 2nd level metrics

    TPR = TP / CondPos            # True Positive Rate / Sensitivity / Hit Rate / Recall
    TNR = TN / CondNeg            # True Negative Rate / Specificity
    PPV = TP / PredPos            # Positive Predictive Value / Precision
    NPV = TN / PredNeg            # Negative Predictive Value
    FPR = FP / CondNeg            # False Positive Rate / Fall-out
    FDR = FP / PredNeg            # False Discovery Rate
    FNR = FN / CondNeg            # False Negative Rate
    FOR = FN / PredNeg            # False Omission Rate

    # 3rd level metrics

    Accuracy = (TP + TN) / Population                  # Accuracy Rate
    PosLR = TPR / FPR                                  # Positive Likelihood Ratio
    NegLR = FNR / TNR                                  # Negative Likelihood Ratio
    DiagOR = PosLR / NegLR                             # Diagnostic Odds Ratio

    # F Scores i.e. F_β = ((1 + β^2) * PPV * TPR)/ (β^2 * PPV + TPR)
    F_1 = (2 * PPV * TPR) / (PPV + TPR)                # F1 Score / Harmonic Mean of Precision and Recall
    F_2 = (5 * PPV * TPR) / (4 * PPV + TPR)            # F2 Score - weights recall higher than precision
    F_05 = (1.25 * PPV * TPR) / (0.25 * PPV + TPR)     # F0.5 Score - weights precision higher than recall

    # Matthews's Correlation Coefficient
    # Guideline:
    # +1 = perfect prediction, 0 = no better than random, -1 = total disagreement
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    # Kappa Coefficient - compares Observed Accuracy with Expected Accuracy (random chance)
    # Guidelines:
    # L&K: <0 = no agreement, 0-0.20 = slight, 0.21-0.40 = fair, 0.41-0.60 = moderate, 0.61-0.80 = substantial, 0.81-1 = almost perfect
    # Fleiss: <0.4 = poor, 0.40-0.75 = fair to good, >0.75 = excellent
    ObsAcc = (TP + TN) / Population                                    # Observed Accuracy
    ExpAcc = (CondPos * PredPos + CondNeg * PredNeg) / Population^2    # Expected Accuracy
    Kappa = (ObsAcc - ExpAcc) / (1 - ExpAcc)                           # Kappa

    # Display metrics
    if size == "short"
        @printf "Accuracy:    %f\n" Accuracy
        @printf "F_1:         %f\n" F_1
    elseif size == "medium"
        @printf "Accuracy:    %f\n" Accuracy
        @printf "F_1:         %f\n" F_1
        @printf "F_2:         %f\n" F_2
        @printf "F_05:        %f\n" F_05
        @printf "MCC:         %f\n" MCC
        @printf "Kappa:       %f\n" Kappa
    elseif size == "long"
        @printf "Accuracy:    %f\n" Accuracy
        @printf "F_1:         %f\n" F_1
        @printf "F_2:         %f\n" F_2
        @printf "F_05:        %f\n" F_05
        @printf "MCC:         %f\n" MCC
        @printf "Kappa:       %f\n" Kappa
        @printf "Precision:   %f\n" PPV
        @printf "Recall:      %f\n" TPR
        @printf "Specificity: %f\n" TNR
    end
end

# ROC Curve
