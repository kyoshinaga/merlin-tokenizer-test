include("./src/testJukaiNLP.jl")
using testJukaiNLP
using Merlin

function encodeSentence(t::IdDict, doc::Vector)
    unk, space, lf = t["UNKNOWN"], t[" "], t["\n"]
    chars = Int[]
    ranges = UnitRange{Int}[]
    pos = 1
  
    for (word, tag) in doc
        for c in tag
            # c == '_' && continue
            # c == 'S' && continue
            # if c == 'S' # Space
            # push!(chars, space)
            #elseif c == 'N' # Newline
            if c == 'N'
                push!(chars, lf)
                pos += 1
            end
        end
        for c in word
            push!(chars, t[string(c)])
        end
        tag != 'S' && push!(ranges, pos:pos+length(word) - 1)
        pos += length(word)
    end
    chars, ranges
end

function accuracy(golds:: Vector{Int}, preds::Vector{Int})
  @assert length(golds) == length(preds)
  correct = 0
  total = 0
  for i= 1:length(golds)
    golds[i] == preds[i] && (correct += 1)
    total += 1
  end
  correct , total
end

t = h5loadTokenizer("./model/tokenizer_20161011.h5")
jpnTestDoc = readknp("corpus/950110.KNP")

sentId = 0

outf = open("./data/analysis.tsv", "w")
for sent in jpnTestDoc
    test_x, ranges = encodeSentence(t.dict, sent) 
    test_y = encode(t.tagset, ranges, length(test_x))

    test_z = argmax(t.model(test_x).data, 1)

    correct, total = accuracy(test_y, test_z)

    strVec = Vector{Char}(join(map(r -> r[1], sent), ""))

    if correct != total
        write(outf, "$(sentId)\n")
        for r in zip(strVec, test_y, test_z)
            outstring = join(r,"\t")
            write(outf, "$(outstring)\n")
        end
    end
    sentId += 1
end

close(outf)
