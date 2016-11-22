include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []
correct = []

biTags = Dict(
"I_Ba"=> 1,
"I_Ia"=> 2,
"I_B"=>  3,
"I_I"=>  4,
"E_Ba"=> 5,
"E_Ia"=> 6,
"E_B"=>  7,
"E_I"=>  8,
"O_Ba"=> 9,
"O_Ia"=>10,
"O_B"=> 11,
"O_I"=> 12
)

push!(correct, biTags["O_Ba"])
push!(sent, ["ナルト", "_N", "B"])
push!(correct, biTags["I_B"])
push!(correct, biTags["I_B"])
push!(correct, biTags["E_B"])
push!(sent, ["激闘", "_", "I"])
push!(correct, biTags["I_I"])
push!(correct, biTags["E_I"])
push!(sent, ["忍者", "_", "Ia"])
push!(correct, biTags["I_Ia"])
push!(correct, biTags["E_Ia"])
push!(sent, ["対戦", "_", "I"])
push!(correct, biTags["I_I"])
push!(correct, biTags["E_I"])
push!(sent, ["三","_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["の", "_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["キャラクター", "_", "Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["は", "_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["全部", "_", "B"])
push!(correct, biTags["I_B"])
push!(correct, biTags["E_B"])
push!(sent, ["で", "_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["何", "_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["体", "_", "I"])
push!(correct, biTags["E_I"])
push!(sent, ["居る", "_", "B"])
push!(correct, biTags["I_B"])
push!(correct, biTags["E_B"])
push!(sent, ["の", "_", "B"])
push!(correct, biTags["E_B"])
push!(sent, ["です", "_", "Ia"])
push!(correct, biTags["I_Ia"])
push!(correct, biTags["E_Ia"])
push!(sent, ["か", "_", "Ba"])
push!(correct, biTags["E_Ba"])
push!(doc, sent)

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
