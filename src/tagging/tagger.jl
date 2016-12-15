using HDF5

export Tagger, h5convert

type Tagger <: Functor
	prefix::String
	wordict::IntDict
	chardict::IntDict{String}
	tagdict::IntDict{String}
	model::Model
end

function Tagger(trainCorpus::String, validCorpus::String ;
	prefix::String = "",
	wordEmbDim=64, wordWindow = 5,
	charEmbDim=32, charWindow = 5)

	trainData, trainWords = readCorpus(trainCorpus)
	validData, validWords = readCorpus(validCorpus)

	words = concat(1, Var(trainWords), Var(validWords)).data

	wordEmbeds = Lookup(Float32, 100000, wordEmbDim)
	charEmbeds = Lookup(Float32, 10000,  charEmbDim)

	worddict = IntDict(words)
	chardict = IntDict{String}()
	tagdict = IntDict{String}()

	train_x, train_y = encode(trainData, worddict, chardict, tagdict, true)
	valid_x, valid_y = encode(validData, worddict, chardict, tagdict, true)

	model = Model(wordembeds, charembeds, length(tagdict))

	Tagger(prefix, worddict, chardict, tagdict, model), train_x, train_y, valid_x, valid_y
end

function readCorpus(path::String)
	dict = h5read(path, "Merlin")
	delete!(dict, "#TYPE")
	doc = []
	word = []

	push!(word, "UNKNOWN")
	push!(word, " ")
	push!(word, "\n")

	for i = 1:length(dict)
		s = dict[string(i)]
		sent = []
		delete!(s, "#TYPE")
		for j = 1:length(s)
			token = s[string(j)]
			if(length(token[1]) > 0)
				push!(sent, token)
				push!(word, token[1])
			endend
		end
		(length(sent) > 0) && (push!(doc, sent))
	end
	doc, word
end

function encode(data::Vector, wordict, chardict, tagdict, append::Bool)

	data_x, data_y = Vector{Token}[], Vector{Int}[]
	unkword = wordict["UNKNOWN"]

	for sent in data
		push!(data_x, Token[])
		push!(data_y, Int[])
		for items in sent
			word, tag = items[1], items[2]
			wordid = get(worddict, word, unkword)

			chars = Vector{Char}(word)
			if append
				charids = map(c -> push!(chardict, string(c)), chars)
			elseif
				charids = map(c -> get(chardict, string(c), 0), chars)
			end
			(length(charids) == 0) && println(string("word: ", word," tag: ", tag))
			tagid = push!(tagdict, tag)
			token = Token(wordid, charids)
			push!(data_x[end], token)
			push!(data_y[end], tagid)
		end
	end
	data_x, data_y
end

function h5convert(x::Tagger)
end
#function (t::Tagger)()
