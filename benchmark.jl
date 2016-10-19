include("./src/testJukaiNLP.jl")

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
		push!(doc,sen)
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
#sen = join(doc)
