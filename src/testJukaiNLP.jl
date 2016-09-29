module testJukaiNLP

#using JuCUDA
#using JuCUDNN
using Merlin
using Merlin: Graph, GraphNode, Embedding
using HDF5

import Merlin: h5save, h5writedict, h5dict, h5convert

abstract Functor

export Tokenizer, train, convTest, im2col, h5convert
#export Tokenizer, TokenizerCuda, train

include("testRead.jl")
include("testIddict.jl")
include("testTagset.jl")
include("testTokenizer.jl")
include("testTrain.jl")
include("testConv.jl")
include("hdf5.jl")

end
