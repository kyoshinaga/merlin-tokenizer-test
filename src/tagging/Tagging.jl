module tagging
importall ..bioTagging

using Merlin
using HDF5

export train, Tagger

abstract Functor

include("intdict.jl")
include("model.jl")
inlcude("token.jl")
include("tagger.jl")
inlcude("train.jl")

end
