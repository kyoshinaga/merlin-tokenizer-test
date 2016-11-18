include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []
correct = []
biTags = Dict("Ba"=>1,"Ia"=>2,"B"=>3,"I"=>4)

push!(sent, ["ナルト", "B"])
push!(sent, ["激闘","I"])
push!(sent, ["忍者","Ia"])
push!(sent, ["対戦","I"])
push!(sent, ["三","Ba"])
push!(sent, ["の","Ba"])
push!(sent, ["キャラクター","Ba"])
push!(sent, ["は","Ba"])
push!(sent, ["全部","B"])
push!(sent, ["で","Ba"])
push!(sent, ["何","Ba"])
push!(sent, ["体","I"])
push!(sent, ["居る","B"])
push!(sent, ["の","B"])
push!(sent, ["です","Ia"])
push!(sent, ["か","Ba"])
push!(doc, sent)

correct = map(sent) do x
	biTags(x[2])
end

t = Tokenizer("")
words, tags = encode(t, doc)

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
