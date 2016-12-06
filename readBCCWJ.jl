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

map(fileList[124:124]) do f
	push!(doneList, f)
	println(string(f,",$(length(doneList)):$(numList)"))
	suw, luw = readBCCWJ(string(prefix, f))
	push!(doc, suw)
	push!(docLuw, luw)
end

doc = flattenDoc(doc)
docLuw = flattenDoc(docLuw)

#numOfData = length(docLuw)
#numOfTrainData = Int(floor(0.8 * numOfData))
#numOfValidData = Int(floor(0.9 * numOfData))
#pickItemList = randperm(numOfData)
#jpnTrainDoc = copy(docLuw[pickItemList[1:numOfTrainData]])
#jpnValidDoc = copy(docLuw[pickItemList[(numOfTrainData+1):numOfValidData]])
#jpnTestDoc = copy(docLuw[pickItemList[(numOfValidData+1):numOfData]])

#Merlin.h5save("/data/kyoshinaga/corpus/sampleDoc.h5",docLuw)
#Merlin.h5save("./corpus/jpnValidDoc.h5",jpnValidDoc)
#Merlin.h5save("./corpus/jpnTestDoc.h5",jpnTestDoc)
