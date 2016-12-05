include("src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin: Graph, GraphNode, Embedding
using HDF5

using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

doc = []
docLuw = []

prefix = "./corpus/bccwj/"
fileList = readstring(`ls $(prefix)`)
fileList = split(chomp(fileList),'\n')
numList = length(fileList)
doneList = []
index = 0

map(fileList[10:10]) do f
	push!(doneList, f)
	println(string(f,",$(length(doneList)):$(numList)"))
	suw, luw = readBCCWJ(string(prefix, f))
	push!(doc, suw)
	push!(docLuw, luw)
end

doc = flattenDoc(doc)
docLuw = flattenDoc(docLuw)

docCombine = []

for (ioe, bi) in zip(doc, docLuw)
	v = []
	for (w1, w2) in zip(ioe, bi)
		push!(v,[w2[1], w1[2], w2[2]])
	end
	push!(docCombine, v)
end

function doTest(trainData, validData, prefix::String, nepoch::Int, emboutCh::Int, convFilterWidth::Int;learningRate=0.0001, dynamicRate::Bool=false)

    success(`mkdir -p ./data/$(prefix)`)
    success(`mkdir -p ./model/$(prefix)`)

    t = Tokenizer(prefix,emboutCh=emboutCh,convFilterWidth=convFilterWidth)

    beginTime = time()
    @time @CPUtime train(t, nepoch, trainData, validData, learningRate=learningRate,dynamicRate=dynamicRate)
    endTime = time()

    outf = open("./data/$(prefix)/computeTime.txt","w")
    write(outf,"time:\t$(endTime - beginTime)")
    close(outf)

    h5save(string("./model/",prefix,"/tokenizer_result.h5"),t)
end

nepoch = 2000
embCh = 32

prefix = "20161124/test"
doTest(docCombine, docCombine, prefix, nepoch, embCh, 9)

#numOfData = length(docCombine)
#numOfTrainData = Int(floor(0.8 * numOfData))
#numOfValidData = Int(floor(0.9 * numOfData))
#pickItemList = randperm(numOfData)
#jpnTrainDoc = copy(docCombine[pickItemList[1:numOfTrainData]])
#jpnValidDoc = copy(docCombine[pickItemList[(numOfTrainData+1):numOfValidData]])
#jpnTestDoc  = copy(docCombine[pickItemList[(numOfValidData+1):numOfData]])
#
#Merlin.h5save("./corpus/jpnTrainDoc.h5",jpnTrainDoc)
#Merlin.h5save("./corpus/jpnValidDoc.h5",jpnValidDoc)
#Merlin.h5save("./corpus/jpnTestDoc.h5",jpnTestDoc)
