include("./src/testJukaiNLP.jl")

using testJukaiNLP
using Merlin

function readKNPSentence(path::String)
  doc = []
  lines = open(readlines,path)
  comment = Char['*','#']
  sen = []

  for line in lines
    if startswith(line, comment)
      continue
    end
    line = chomp(line)
    if line == "EOS"
		push!(doc,join(sen)	)
  		sen = []
        continue
    end
    items = split(line, ' ')
    strVector = Vector{Char}(items[1])
    append!(sen, strVector)
  end
  doc
end

doc = readKNPSentence("./corpus/950110.KNP")
docGold = readknp("./corpus/950110.KNP")

t = h5loadTokenizer("./model/tokenizer_knp.h5","test.tsv")

# Gold
charsGold, rangesGold = encode(t, docGold)
tagsGold = encode(t.tagset, ragesGold, length(charsGold))

# model
tagsTest = t.model(charsGold)
tagsTest = argmax(tagsText.data, 1)

# mecab
mecab = String[]
for s in doc
	r = readstring(pipeline(`echo $s`, `mecab -d /home/kyoshinaga/local/lib/mecab/dic/jumandic/`))
	r = chomp(r)
	r = split(r, '\n')
	map(r) do x
		push!(mecab,replace(x, r"\t|\,",' '))
	end
end
docMecab = readMecabJuman(mecab)
charsMecab, rangesMecab = encode(t, docMecab)
tagsMecab = encode(t.tagset, rangesMecab, length(charsMecab))
