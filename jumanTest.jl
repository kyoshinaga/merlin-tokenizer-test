include("src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin: Graph, GraphNode, Embedding
using HDF5

import Merlin: h5save, h5writedict, h5dict, h5convert

prefix = "./corpus"
dirList = readstring(`ls $(prefix)`)
dirList = split(chomp(dirList),'\n')
numList = length(dirList)
doneList = String[]

doc = []

map(dirList) do dir
    fileList = []
    fileList = readstring(`ls $(prefix)/$(dir)`)
    fileList = split(chomp(fileList), '\n')
    numFile = length(fileList)
    map(fileList) do file
        push!(doc, readJuman("./$(prefix)/$(dir)/$(file)"))
    end
end

doc = flattenDoc(doc)

numOfData = length(doc)
numOfTrainData = Int(floor(0.9 * numOfData))
pickItemList = randperm(numOfData)
jpnTrainDoc = copy(doc[pickItemList[1:numOfTrainData]])
jpnTestDoc = copy(doc[pickItemList[(numOfTrainData+1):numOfData]])

prefix = "tokenizer_20161023_Juman"

t = Tokenizer(string("./log/trainProgress_",prefix,".tsv"))

@time train(t, 100, jpnTrainDoc, jpnTestDoc)

h5save(string("./model/", prefix, ".h5"),t)
