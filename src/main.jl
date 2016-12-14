using Merlin
using MLDatasets
using HDF5

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
            end
        end
        (length(sent) > 0) && (push!(doc, sent))
    end
    doc, word
end

function main()
    traindata, words = readCorpus("../corpus/sampleDoc.h5")

#    h5file = "wordembeds_nyt100.h5"
#    words = h5read(h5file, "s")
#    wordembeds = Lookup(h5read(h5file,"v"))
    wordembeds = Lookup(Float32, 50000, 100)
    charembeds = Lookup(Float32, 10000, 10)

    worddict = IntDict(words)
    chardict = IntDict{String}()
    tagdict = IntDict{String}()

#    testdata = UD_English.testdata()
#    traindata = UD_English.traindata()
#    testdata = UD_English.testdata()
    info("# sentences of train data: $(length(traindata))")
#    info("# sentences of test data: $(length(testdata))")

    train_x, train_y = encode(traindata, worddict, chardict, tagdict, true)
#    test_x, test_y = encode(testdata, worddict, chardict, tagdict, false)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")

    model = Model(wordembeds, charembeds, length(tagdict))
    # model = Merlin.load("postagger.h5", "model")
    train(500, model, train_x, train_y, train_x, train_y)

    #Merlin.save("postagger.h5", "w", "model", model)
    train_x, train_y
end

function train(nepochs::Int, model, train_x, train_y, test_x, test_y)
    opt = SGD()
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = map(x -> predict(model,x), test_x)
        acc = accuracy(test_y, test_z)
        println("test acc.: $(acc)")
        println("")
    end
end

predict(model, data) = argmax(model(data).data, 1)

function encode(data::Vector, worddict, chardict, tagdict, append::Bool)
    data_x, data_y = Vector{Token}[], Vector{Int}[]
    unkword = worddict["UNKNOWN"]
    for sent in data
        push!(data_x, Token[])
        push!(data_y, Int[])
        for items in sent
            word, tag = items[1], items[2]
            #word0 = replace(word, r"[0-9]", '0')
            #wordid = get(worddict, lowercase(word0), unkword)
            wordid = get(worddict, word, unkword)

            chars = Vector{Char}(word)
            if append
                charids = map(c -> push!(chardict,string(c)), chars)
            else
                charids = map(c -> get(chardict,string(c),0), chars)
            end
            (length(charids) == 0) && println(string("word: ",word," tag: ", tag))
            tagid = push!(tagdict, tag)
            token = Token(wordid, charids)
            push!(data_x[end], token)
            push!(data_y[end], tagid)
        end
    end
    data_x, data_y
end

function accuracy(golds::Vector{Vector{Int}}, preds::Vector{Vector{Int}})
    @assert length(golds) == length(preds)
    correct = 0
    total = 0
    for i = 1:length(golds)
        @assert length(golds[i]) == length(preds[i])
        for j = 1:length(golds[i])
            golds[i][j] == preds[i][j] && (correct += 1)
            total += 1
        end
    end
    correct / total
end

include("intdict.jl")
include("token.jl")
include("model.jl")

train_x, train_y = main()
