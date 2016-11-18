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

#numOfData = length(doc)
#numOfTrainData = Int(floor(0.8 * numOfData))
#numOfValidData = Int(floor(0.9 * numOfData))
#pickItemList = randperm(numOfData)
#jpnTrainDoc = copy(doc[pickItemList[1:numOfTrainData]])
#jpnValidDoc = copy(doc[pickItemList[(numOfTrainData+1):numOfValidData]])
#jpnTestDoc = copy(doc[pickItemList[(numOfValidData+1):numOfData]])
#
#Merlin.h5save("./corpus/jpnTrainDoc.h5",jpnTrainDoc)
#Merlin.h5save("./corpus/jpnValidDoc.h5",jpnValidDoc)
#Merlin.h5save("./corpus/jpnTestDoc.h5",jpnTestDoc)

#jpnTrainDoc = readCorpus("./corpus/jpnTrainDoc.h5")
#jpnValidDoc = readCorpus("./corpus/jpnValidDoc.h5")
#jpnTestDoc = readCorpus("./corpus/jpnTestDoc.h5")
#
#println("Train data:\t$(length(jpnTrainDoc))")
#println("Valid data:\t$(length(jpnValidDoc))")
#println("Test data:\t$(length(jpnTestDoc))")
#
#prefix = "tokenizer_20161026_KNP_ver2"
#
#t = Tokenizer(string("./data/",prefix,".tsv"))
#
#@time @CPUtime train(t, 10000, jpnTrainDoc, jpnValidDoc)
#
#h5save(string("./model/", prefix, ".h5"),t)
