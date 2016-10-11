include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []

push!(sent, ["村山", "_"])
push!(sent, ["富一", "_"])
push!(sent, ["首相", "_"])
push!(sent, ["は", "_"])
push!(sent, ["、", "S"])
push!(sent, ["決めた", "_"])
push!(sent, ["。", "S"])

push!(doc, sent)

sent = []
push!(sent, ["大蔵", "N"])
push!(sent, ["省", "_"])
push!(sent, ["。", "S"])

push!(doc, sent)

correct = [1,3,1,3,1,3,3,2,1,1,3,2,2,1,3,3,2]

t = Tokenizer("")
chars, ranges = encode(t, doc)
tags = encode(t.tagset, ranges, length(chars))

failed = 0
for p in zip(tags, correct)
    p[1] != p[2] && (failed += 1)
end

if failed > 0
    println("===============")
    println("Failed encoding")
    println("===============")
end
