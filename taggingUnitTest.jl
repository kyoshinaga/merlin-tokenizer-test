include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []
correct = []
biTags = Dict("Ba"=>1,"Ia"=>2,"B"=>3,"I"=>4)

push!(sent, ["ナルト", "_","B"])
push!(sent, ["激闘","_","I"])
push!(sent, ["忍者","_","Ia"])
push!(sent, ["対戦","_","I"])
push!(sent, ["三","_","Ba"])
push!(sent, ["の","_","Ba"])
push!(sent, ["キャラクター","_","Ba"])
push!(sent, ["は","_","Ba"])
push!(sent, ["全部","_","B"])
push!(sent, ["で","_","Ba"])
push!(sent, ["何","_","Ba"])
push!(sent, ["体","_","I"])
push!(sent, ["居る","_","B"])
push!(sent, ["の","_","B"])
push!(sent, ["です","_","Ia"])
push!(sent, ["か","_","Ba"])
push!(doc, sent)

correct = map(sent) do x
	biTags[x[2]]
end

t = Tokenizer("")
words, tags = encode(t, doc)

println("correct")
println(transpose(correct))
println("char")
println(transpose(tags[1]))

failed = 0
for p in zip(tags[1], correct)
    p[1] != p[2] && (failed += 1)
end

if failed > 0
    println("===============")
    println("Failed encoding")
    println("===============")
end
