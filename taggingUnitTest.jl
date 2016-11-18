include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []
correct = []

push!(sent, ["村山", "_"])
push!(correct,1)
push!(correct,3)
push!(sent, ["富一", "_"])
push!(correct,1)
push!(correct,3)
push!(sent, ["首相", "_"])
push!(correct,1)
push!(correct,3)
push!(sent, ["は", "_"])
push!(correct,3)
push!(sent, ["、", "S"])
push!(correct,2)
push!(sent, ["決めた", "_"])
push!(correct,1)
push!(correct,1)
push!(correct,3)
push!(sent, ["。", "S"])
push!(correct,2)
push!(doc, sent)

t = Tokenizer("")
chars, ranges = encode(t, doc)
tags = encode(t.tagset, ranges, length(chars))

println("correct")
println(transpose(correct))
println("char")
println(transpose(tags))

failed = 0
for p in zip(tags, correct)
    p[1] != p[2] && (failed += 1)
end

if failed > 0
    println("===============")
    println("Failed encoding")
    println("===============")
end
