include("./src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin
using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

function doTest(trainData, validData, prefix::String, nepoch::Int, embOUtCh::Int, convFilterWidth::Int;learningRate=0.00001, dynamicRate::Bool=false)

    success(`mkdir -p ./data/$(prefix)`)
    success(`mkdir -p ./model/$(prefix)`)

    t = Tokenizer(prefix,emboutCh,convFilterWidth)

    beginTime = time()
    @time @CPUtime train(t, enpoch, trainDoc, validDoc, learningRate=learningRate,dynamicRate=dynamicRate)
    endTime = time()

    outf = open("./data/$(prefix)/computeTime.txt","w")
    write(outf,"time:\t$(endTime - beginTime)") 
    close(outf)

    h5save(string("./model/",prefix,"/tokenizer_result.h5"),t)
end

jpnTrainDoc = readCorpus("./corpus/jpnTrainDoc.h5")
jpnValidDoc = readCorpus("./corpus/jpnValidDoc.h5")

println("Train data:\t($(length(jpnTrainDoc)))")
println("Valid data:\t($(length(jpnValidDoc)))")

# pattern1
doTest(jpnTrainDoc,jpnValidDoc,"20161114/pattern1", 200, 32, 7)
