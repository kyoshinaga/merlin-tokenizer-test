include("./src/testJukaiNLP.jl")

using testJukaiNLP
using Merlin

docGold = readCorpus("./corpus/jpnTestDoc.h5")

# delete \t
docGoldFormatted = []
map(docGold) do x
    sent = []
    map(x) do word
        (contains(word[1], "\t")) || push!(sent, word)
    end
    push!(docGoldFormatted, sent)
end

println("test length: $(length(docGoldFormatted))")
doc = map(x -> join(map(r -> r[1], x)), docGoldFormatted)

t = h5loadTokenizer("./model/20161114/pattern1/tokenizer_result.h5")

# Gold
charsGold, rangesGold = encode(t, docGoldFormatted)
tagsGold = []
map(zip(charsGold, rangesGold)) do x
    push!(tagsGold, encode(t.tagset, x[2], length(x[1])))
end

# model
tagsTest = map(charsGold) do x
    argmax(t.model(x).data, 1)
end

# mecab
mecab = map(doc) do x
    r = readstring(pipeline(`echo $x`, `mecab -d /usr/local/lib/mecab/dic/jumandic/`))
	r = chomp(r)
	r = split(r, '\n')
    r = map(y -> string(replace(y, r"\t|\,", ' ')), r)
    r
end
docMecab = map(x -> testJukaiNLP.readMecabJuman(x), mecab)
charsMecab, rangesMecab = encode(t, docMecab)
tagsMecab = []
map(zip(charsMecab, rangesMecab)) do x
    push!(tagsMecab, encode(t.tagset, x[2], length(x[1])))
end

#totalGold = 0
#correctTest = 0
##correctMecab = 0
##
#for i = 1:length(tagsGold)
#    for j = 1:length(tagsGold[i])
#        tagsGold[i][j] == tagsTest[i][j] && (correctTest += 1)
##        tagsGold[i][j] == tagsMecab[i][j] && (correctMecab += 1)
#        totalGold += 1
#    end
#end
##
##println("Mecab acc.: $(correctMecab / totalGold)")
#println("Test acc.: $(correctTest / totalGold)")
