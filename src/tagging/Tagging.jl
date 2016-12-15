module tagging
importall ..bioTagging

using Merlin
using HDF5

export train, Tagger

abstract Functor

include("token.jl")
include("intdict.jl")
include("model.jl")
include("tagger.jl")
include("train.jl")

end
