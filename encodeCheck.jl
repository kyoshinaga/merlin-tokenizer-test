include("src/testJukaiNLP.jl")
using testJukaiNLP

using Merlin: Graph, GraphNode, Embedding
using HDF5

using CPUTime

import Merlin: h5save, h5writedict, h5dict, h5convert

jpnTrainDoc = []
jpnTestDoc = []

push!(jpnTrainDoc,readknp("corpus/950101.KNP"))
push!(jpnTrainDoc,readknp("corpus/950103.KNP"))
push!(jpnTrainDoc,readknp("corpus/950104.KNP"))
push!(jpnTrainDoc,readknp("corpus/950105.KNP"))
push!(jpnTrainDoc,readknp("corpus/950106.KNP"))
push!(jpnTrainDoc,readknp("corpus/950107.KNP"))
push!(jpnTrainDoc,readknp("corpus/950108.KNP"))
push!(jpnTrainDoc,readknp("corpus/950109.KNP"))
push!(jpnTestDoc,readknp("corpus/950110.KNP"))
push!(jpnTrainDoc,readknp("corpus/950111.KNP"))
push!(jpnTrainDoc,readknp("corpus/950112.KNP"))
push!(jpnTrainDoc,readknp("corpus/950113.KNP"))
push!(jpnTrainDoc,readknp("corpus/950114.KNP"))
push!(jpnTrainDoc,readknp("corpus/950115.KNP"))
push!(jpnTrainDoc,readknp("corpus/950116.KNP"))
push!(jpnTrainDoc,readknp("corpus/950117.KNP"))

push!(jpnTrainDoc, readBCCWJ("corpus/PB10_00047.xml"))

jpnTrainDoc = flattenDoc(jpnTrainDoc)
jpnTestDoc = flattenDoc(jpnTestDoc)

t = Tokenizer("BCCWJ_test")
