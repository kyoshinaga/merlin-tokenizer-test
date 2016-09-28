module testJukaiNLP

#using JuCUDA
#using JuCUDNN
using Merlin
using HDF5

abstract Functor

export Tokenizer, train
#export Tokenizer, TokenizerCuda, train

include("testRead.jl")
include("testIddict.jl")
include("testTagset.jl")
include("testTokenizer.jl")
include("testTrain.jl")
include("hdf5.jl")

end
