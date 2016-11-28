include("./src/testJukaiNLP.jl")
using testJukaiNLP

doc = []
sent = []
corrects = []
correct = []

biTags = Dict(
"I_Ba"=> 1,
"I_Ia"=> 2,
"I_B"=>  3,
"I_I"=>  4,
"O_Ba"=> 5,
"O_Ia"=> 6,
"O_B"=>  7,
"O_I"=>  8,
"E_Ba"=> 9,
"E_Ia"=>10,
"E_B"=> 11,
"E_I"=> 12
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
push!(corrects, correct)

sent = []
correct = []

push!(correct, biTags["O_Ba"])
push!(sent, ["↑", "SN", "Ba"])
push!(correct, biTags["O_Ba"])
push!(sent, ["神々", "_", "Ba"])
push!(correct, biTags["I_Ba"])
push!(correct, biTags["E_Ba"])
push!(sent, ["の", "_", "I"])
push!(correct, biTags["E_I"])
push!(sent, ["零落", "_", "Ia"])
push!(correct, biTags["I_Ia"])
push!(correct, biTags["E_Ia"])
push!(sent, ["。", "S", "Ba"])
push!(correct, biTags["O_Ba"])
push!(doc, sent)
push!(corrects, correct)

t = Tokenizer("")
words, tags = encode(t, doc)

println("correct[1]")
println(transpose(corrects[1]))
println("char[1]")
println(transpose(tags[1]))

println("correct[2]")
println(transpose(corrects[2]))
println("char[2]")
println(transpose(tags[2]))

failed = 0
for (tag, cor) in zip(tags, correct)
	for (t, c) in zip(tag, cor)
		t != c && (failed += 1)
	end
end

if failed > 0
    println("===============")
    println("Failed encoding")
    println("===============")
end
