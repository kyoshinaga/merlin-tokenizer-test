include("src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin: Graph, GraphNode, Embedding
using HDF5

using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

#doc = []
#
#push!(doc,readknp("corpus/950101.KNP"))
#push!(doc,readknp("corpus/950103.KNP"))
#push!(doc,readknp("corpus/950104.KNP"))
#push!(doc,readknp("corpus/950105.KNP"))
#push!(doc,readknp("corpus/950106.KNP"))
#push!(doc,readknp("corpus/950107.KNP"))
#push!(doc,readknp("corpus/950108.KNP"))
#push!(doc,readknp("corpus/950109.KNP"))
#push!(doc,readknp("corpus/950110.KNP"))
#push!(doc,readknp("corpus/950111.KNP"))
#push!(doc,readknp("corpus/950112.KNP"))
#push!(doc,readknp("corpus/950113.KNP"))
#push!(doc,readknp("corpus/950114.KNP"))
#push!(doc,readknp("corpus/950115.KNP"))
#push!(doc,readknp("corpus/950116.KNP"))
#push!(doc,readknp("corpus/950117.KNP"))

#prefix = "./corpus/bccwj/"
#fileList = readstring(`ls $(prefix)`)
#fileList = split(chomp(fileList),'\n')
#numList = length(fileList)
#doneList = []
#
#map(fileList) do f
#	push!(doneList, f)
#	println(string(f,",$(length(doneList)):$(numList)"))
#	suw, luw = readBCCWJ(string(prefix, f))
#	push!(doc, suw)
#end

#doc = flattenDoc(doc)
#
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

jpnTrainDoc = readCorpus("./corpus/jpnTrainDoc.h5")
jpnValidDoc = readCorpus("./corpus/jpnValidDoc.h5")
jpnTestDoc = readCorpus("./corpus/jpnTestDoc.h5")

println("Train data:\t$(length(jpnTrainDoc))")
println("Valid data:\t$(length(jpnValidDoc))")
println("Test data:\t$(length(jpnTestDoc))")

prefix = "tokenizer_20161026_KNP_ver2"

t = Tokenizer(string("./data/",prefix,".tsv"))

@time @CPUtime train(t, 10000, jpnTrainDoc, jpnValidDoc)

h5save(string("./model/", prefix, ".h5"),t)
