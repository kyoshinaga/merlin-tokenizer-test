export encode, convJukai, flatten, flattenDoc, train

using Formatting

function encode(t::Tokenizer, doc::Vector)
	unk, space, lf = t.dict["UNKNOWN"], t.dict[" "], t.dict["\n"]
	biTag = Dict(
	"I_Ba"=>t.tagset.I_Ba,
	"I_Ia"=>t.tagset.I_Ia,
	"I_B"=> t.tagset.I_B,
	"I_I"=> t.tagset.I_I,
	"E_Ba"=>t.tagset.E_Ba,
	"E_Ia"=>t.tagset.E_Ia,
	"E_B"=> t.tagset.E_B,
	"E_I"=> t.tagset.E_I,
	"O_Ba"=> t.tagset.O_Ba,
	"O_Ia"=> t.tagset.O_Ia,
	"O_B"=> t.tagset.O_B,
	"O_I"=> t.tagset.O_I
	)
	chars = []
	tags = []
	for sent in doc
		charVector = Int[]
		tagVector = Int[]
		pos = 1
		for (word, ioe, bi) in sent
			if endswith(ioe, 'N')
				push!(charVector, lf)
				push!(tagVector, biTag["O_Ba"])
				pos += 1
			end
			if startswith(ioe, 'S')
				push!(charVector, push!(t.dict, string(word)))
				push!(tagVector, biTag[string("O_",bi)])
			else
				for c in word
					push!(charVector, push!(t.dict, string(c)))
					push!(tagVector, biTag[string("I_",bi)])
				end
				pos += length(word)
				tagVector[pos-1] = biTag[string("E_",bi)]
			end
		end
		push!(chars, charVector)
		push!(tags, tagVector)
	end
	chars, tags
end

function train(t::Tokenizer, nepoch::Int, trainData::Vector, testData::Vector; batchFlag=false, dynamicRate=false, learningRate=0.001)
	chars, tags = encode(t, trainData)
	#tags = []
	#map(zip(chars, ranges)) do x
	#	push!(tags, encode(t.tagset, x[2], length(x[1])))
	#end

	if (batchFlag)
		train_x = []
		train_y = []
		push!(train_x, flatten(chars))
		push!(train_y, flatten(tags))
	else
		train_x = chars
		train_y = tags
	end

	chars2, tags2 = encode(t, testData)
	#tags2 = []
	#map(zip(chars2, ranges2)) do x
	#	push!(tags2, encode(t.tagset, x[2], length(x[1])))
	#end
	test_x, test_y = chars2, tags2

	momentumRate = 0.9

	opt = SGD(learningRate, momentum=momentumRate)

	outLR = open(string("./data/",t.prefix,"/learningRate.tsv"),"w")
	write(outLR,"learning rate:\t$(learningRate)\n")
	write(outLR,"momentum rate:\t$(momentumRate)\n")
	close(outLR)

	outf = open(string("./data/",t.prefix,"/trainProgress.tsv"),"w")
	write(outf,"epoch\ttrain gold\ttrain correct\ttrain acc.\ttest gold\ttest correct\ttest acc.\ttrain loss\tvalid loss\n")

	firstUpdatedFlag = true
	secondUpdatedFlag = true

	fmt = "%1.5f"

	for epoch = 1:nepoch
		println("================")
		println("epoch : $(epoch)")
		trainLoss = fit(train_x, train_y, t.model, crossentropy, opt, progress=true)
		println("train loss : $(trainLoss)")

		train_z = map(train_x) do x
			argmax(t.model(x).data, 1)
		end

		test_z = map(test_x) do x
			argmax(t.model(x).data, 1)
		end

		train_correct, train_total = 0, 0
		test_correct, test_total = 0, 0

		map(zip(train_y, train_z)) do x
			correct, total = accuracy(x[1], x[2])
			train_correct += correct
			train_total += total
		end

		map(zip(test_y, test_z)) do x
			correct, total = accuracy(x[1], x[2])
			test_correct += correct
			test_total += total
		end

		trainAcc = sprintf1(fmt, (train_correct / train_total))
		validAcc = sprintf1(fmt, (test_correct / test_total))

		println("Train")
		println("\tGold : $(train_total), Correct: $(train_correct)")
		println("\ttest acc.: $(trainAcc)")
		println("Valid")
		println("\tGold : $(test_total), Correct: $(test_correct)")
		println("\ttest acc.: $(validAcc)")
		println("")

		validLoss = 0
		for data in zip(test_x,test_y)
			z = t.model(data[1])
			l = crossentropy(data[2], z)
			validLoss += sum(l.data)
		end

		# file output
		write(outf, "$(epoch)\t$(train_total)\t$(train_correct)\t$(trainAcc)\t$(test_total)\t$(test_correct)\t$(validAcc)\t$(trainLoss)\t$(validLoss)\n")

		if dynamicRate
			if (epoch > nepoch * 0.2) && firstUpdatedFlag
				learningRate *= Float32(0.1)
				firstUpdatedFlag = false
			end
			if epoch > nepoch * 0.6 && secondUpdatedFlag
				learningRate *= Float32(0.1)
				secondUpdatedFlag = false
			end
		end

		if epoch % (nepoch/10) == 0
			h5save(string("./model/",t.prefix,"/tokenizer_",string(epoch),".h5"),t)
			flush(outf)
		end

	end

	close(outf)
end

function flatten(data::Vector)
	res = Int[]
	for x in data
		append!(res, x)
	end
	res
end

function flattenDoc(data::Vector)
	res = []
	for x in data
		append!(res, x)
	end
	res
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
