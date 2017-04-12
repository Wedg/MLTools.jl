module MLTools

##############################################################################
##
## Dependencies
##
##############################################################################

using DataFrames
using Gadfly
set_default_plot_size(15cm, 15cm)
using Colors

##############################################################################
##
## Exported methods and types
##
##############################################################################

export
    # information
	entropy,
	jointEntropy,
    mutualInformation,

	# normalise
    normalise,
    normalise!,

    # polynomial
    polynomial_features,

    # accuracy
    confusion_matrix,
    plot_confusion_matrix,
    print_binary_accuracy,

    # roc
    plot_ROC_curve

#=
    TODO
    kFoldClassPropnPrep,
    kFoldClassPropnArrays,
    shufflePartition2,
    shufflePartition3,
    propnPartition2,
=#

##############################################################################
##
## Load source files
##
##############################################################################

include("normalise.jl")
include("information.jl")
include("polynomial.jl")
include("accuracy.jl")
include("roc.jl")
include("crossvalidation.jl")

end # module
