include("src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin: Graph, GraphNode, Embedding
using HDF5

using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

#jpnTrainDoc = readCorpus("./corpus/jpnTrainDoc.h5")
jpnValidDoc = readCorpus("./corpus/jpnValidDoc.h5")
#jpnTestDoc = readCorpus("./corpus/jpnTestDoc.h5")

#println("Train data:\t$(length(jpnTrainDoc))")
println("Valid data:\t$(length(jpnValidDoc))")
#println("Test data:\t$(length(jpnTestDoc))")

prefix = "20161101"

run(`mkdir ./data/$(prefix)`)
run(`mkdir ./model/$(prefix)`)

t = Tokenizer(prefix)

@time @CPUtime train(t, 10, jpnValidDoc, jpnValidDoc)

h5save(string("./model/", prefix, "/tokenizer_result.h5"),t)
