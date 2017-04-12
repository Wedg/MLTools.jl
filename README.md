# MLTools.jl
Utilities for Machine Learning

To use, run the command:
```jlcon
Pkg.clone("https://github.com/Wedg/MLTools.jl.git")
```

## ROC Curve
The ROC plot is built with the function:    
- `plot_ROC_curve(y_cond, y_prob)`

  `y_cond` is the "truth" vector with each element either `0` or `1`.  
  `y_prob` is the hypothesis vector with each element a probability in the range `[0, 1]`.
  
 As well as the ROC curve of the predictor, the plot shows the model's accuracy, true positive rate, and false positive rate as well as the summary statistic AUC (Area Under Curve).
  
![](demo/roc.png)
  
  
## Confusion Matrix
![](demo/cmp.png)
