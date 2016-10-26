include("./src/testJukaiNLP.jl")

using testJukaiNLP
using Merlin

#function readKNPSentence(path::String)
#  doc = []
#  lines = open(readlines,path)
#  comment = Char['*','#']
#  sen = []
#
#  for line in lines
#    if startswith(line, comment)
#      continue
#    end
#    line = chomp(line)
#    if line == "EOS"
#		push!(doc,join(sen)	)
#  		sen = []
#        continue
#    end
#    items = split(line, ' ')
#    strVector = Vector{Char}(items[1])
#    append!(sen, strVector)
#  end
#  doc
#end
#
#doc = readKNPSentence("./corpus/950117.KNP")
#docGold = readknp("./corpus/950117.KNP")


docGold = readCorpus("./corpus/jpnTestDoc.h5")
doc = []
for sent in docGold
    s = join(map(r -> r[1], sent))
    if endswith(sent[1][2],"N")
        sent[1][2] = "_"
        # s = join(["\n",s])
    end
    push!(doc, s)
end

t = h5loadTokenizer("./model/tokenizer_20161019_KNP.h5","test.tsv")

# Gold
charsGold, rangesGold = encode(t, docGold)
tagsGold = encode(t.tagset, rangesGold, length(charsGold))

# model
tagsTest = t.model(charsGold)
tagsTest = argmax(tagsTest.data, 1)

# mecab
mecab = String[]
for s in doc
	r = readstring(pipeline(`echo $s`, `mecab -d /home/kyoshinaga/local/lib/mecab/dic/jumandic/`))
	r = chomp(r)
	r = split(r, '\n')
	map(r) do x
		x == "EOS" || push!(mecab,replace(x, r"\t|\,",' '))
	end
end
docMecab = testJukaiNLP.readMecabJuman(mecab)
charsMecab, rangesMecab = encode(t, docMecab)
tagsMecab = encode(t.tagset, rangesMecab, length(charsMecab))

totalGold = 0
correctTest = 0
correctMecab = 0

for i = 1:length(tagsGold)
    tagsGold[i] == tagsTest[i] && (correctTest += 1)
    tagsGold[i] == tagsMecab[i] && (correctMecab += 1)
    totalGold += 1
end

println("Mecab acc.: $(correctMecab / totalGold)")
println("Test acc.: $(correctTest / totalGold)")
