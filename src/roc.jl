# Plot ROC curve
function plot_ROC_curve(y_cond::Vector{Int}, y_prob::Vector{Float64})

    # Fixed counts
    population = length(y_cond)
    cond_pos = sum(y_cond .== 1)
    cond_neg = sum(y_cond .== 0)
    accuracy = round(sum(y_cond .== round(Int, y_prob)) / population, 2)

    # Number of plot points
    points = 1001
    max = points - 1

    # Calculate TPR and FPR for varied discrimination
    TPR, FPR = zeros(points), zeros(points)
    for d = 0:1:max
        TP, FP = 0, 0
        for i=1:population
            if y_prob[i] >= (max - d) / max
                y_cond[i] == 1 ? TP += 1 : FP += 1
            end
        end
        TPR[d + 1] = TP / cond_pos
        FPR[d + 1] = FP / cond_neg
    end

    # Accuracy
    TP, FP = 0, 0
    for i = 1:population
        if y_prob[i] >= 0.5
            y_cond[i] == 1 ? TP += 1 : FP += 1
        end
    end
    TPR1 = TP / cond_pos
    FPR1 = FP / cond_neg
    df1 = DataFrame(x = FPR1, y = TPR1, label = "Accuracy = $accuracy")

    # AUC
    AUC = 0
    for i=2:length(TPR)
        AUC += (TPR[i] + TPR[i - 1]) * 0.5 * (FPR[i] - FPR[i - 1])
    end
    AUC = round(AUC, 2)
    df2 = DataFrame(x = [0.65], y = [0.3], label = "AUC = $AUC")

    Sigma = Char(931)

    # Plot ROC curve
    plot(
         layer(df1, x="x", y="y", label="label", Geom.point, Geom.label,
               Stat.xticks(ticks=[FPR1]), Stat.yticks(ticks=[TPR1]),
               Theme(default_color=colorant"red", default_point_size=3pt)),
         layer(x=FPR, y=TPR, Geom.step),
         layer(x=[0.0; 1.0], y=[0.0; 1.0], Geom.line,
               Stat.xticks(ticks=[0.0, 0.5, 1.0]), Stat.yticks(ticks=[0.0, 0.5, 1.0]),
               Theme(default_color=colorant"grey", line_style=Gadfly.get_stroke_vector(:dash))),
         layer(df2, x="x", y="y", label="label", Geom.label),
         Scale.x_continuous(minvalue=0., maxvalue=1., labels=x -> @sprintf("%0.2f", x)),
         Scale.y_continuous(minvalue=0., maxvalue=1., labels=y -> @sprintf("%0.2f", y)),
         Guide.ylabel("<b>True Positive Rate </b>\n<br><i>($Sigma True Positives / $Sigma Condition Positive)</i>"),
         Guide.xlabel("<b>False Positive Rate </b>\n<br><i>($Sigma False Positives / $Sigma Condition Negative)</i>")
         )
end
