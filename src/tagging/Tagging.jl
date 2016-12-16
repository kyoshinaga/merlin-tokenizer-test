module tagging
importall ..bioTagging

using Merlin
using HDF5

import Merlin:save

export train, Tagger
export IntDict

abstract Functor

include("token.jl")
include("intdict.jl")
include("model.jl")
include("tagger.jl")
include("train.jl")

end
