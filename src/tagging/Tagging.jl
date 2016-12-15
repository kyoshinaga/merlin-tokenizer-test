module tagging
importall ..bioTagging

using Merlin
using HDF5

export train, Tagger

abstract Functor

inlcude("token.jl")
include("intdict.jl")
include("model.jl")
include("tagger.jl")
inlcude("train.jl")

end
