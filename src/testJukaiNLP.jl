module testJukaiNLP

#using JuCUDA
#using JuCUDNN
using Merlin
# using Merlin: Graph, GraphNode, Embedding
using HDF5

import Merlin: h5load, h5load!

export h5load!,h5load

abstract Functor

include("hdf5.jl")
include("testRead.jl")
include("testIddict.jl")
include("testTagset.jl")
include("Tokenizer.jl")
include("testTrain.jl")

end
