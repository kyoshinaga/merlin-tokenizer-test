module testJukaiNLP

using Merlin

export Tokenizer, train

include("testRead.jl")
include("testIddict.jl")
include("testTagset.jl")
include("testTokenizer.jl")
include("testTrain.jl")

end
