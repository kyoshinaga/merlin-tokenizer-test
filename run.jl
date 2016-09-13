include("src/testJukaiNLP.jl")
using testJukaiNLP

#doc = readconll("corpus/mini-mini-training-set.conll.org",[2,11])
engdoc = readconll("corpus/mini-training-set.conll",[2,11])
jpnTrainDoc = readknp("corpus/950101.KNP.org")
jpnTestDoc = readknp("corpus/950103.KNP")
t = Tokenizer()
