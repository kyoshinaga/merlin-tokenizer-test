include("./src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin
using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

jpnTrainDoc = readCorpus("./corpus/jpnTrainDoc.h5")
jpnValidDoc = readCorpus("./corpus/jpnValidDoc.h5")

println("Train data:\t($(length(jpnTrainDoc)))")
println("Valid data:\t($(length(jpnValidDoc)))")

prefix = "pattern12"

success(`mkdir -p ./data/$(prefix)`)
success(`mkdir -p ./model/$(prefix)`)

t = Tokenizer(prefix)

beginTime = time()
@time @CPUtime train(t, 500, jpnTrainDoc, jpnValidDoc)
endTime = time()

outf = open("./data/$(prefix)/computeTime.txt","w")
write(outf,"time:\t$(endTime - beginTime)") 
close(outf)

h5save(string("./model/",prefix,"/tokenizer_result.h5"),t)
