export encode, convJukai, flatten, flattenDoc, train

function encode(t::Tokenizer, doc::Vector)
	unk, space, lf = t.dict["UNKNOWN"], t.dict[" "], t.dict["\n"]
	chars = []
	ranges = []
	for sent in doc
		charVector = Int[]
		rangeVector = UnitRange{Int}[]
		pos = 1
		for (word, tag) in sent
			if endswith(tag,'N')
				push!(charVector, lf)
				pos += 1
			end
			for c in word
				push!(charVector, push!(t.dict, string(c)))
			end
			startswith(tag,'S') || push!(rangeVector, pos:pos+length(word) - 1)
			pos += length(word)
		end
		push!(chars, charVector)
		push!(ranges, rangeVector)
	end
	chars, ranges
end

function train(t::Tokenizer, nepoch::Int, trainData::Vector, testData::Vector; batchFlag=false)
  chars, ranges = encode(t, trainData)
  tags = []
  map(zip(chars, ranges)) do x
	  push!(tags, encode(t.tagset, x[2], length(x[1])))
  end

  if (batchFlag)
    train_x = []
    train_y = []
    push!(train_x, flatten(chars))
    push!(train_y, flatten(tags))
  else
    train_x = chars
    train_y = tags
  end

  chars2, ranges2 = encode(t, testData)
  tags2 = []
  map(zip(chars2, ranges2)) do x
	  push!(tags2, encode(t.tagset, x[2], length(x[1])))
  end
  test_x, test_y = chars2, tags2

  learningRate = 0.00001
  momentumRate = 0.9

  opt = SGD(learningRate, momentum=momentumRate)

  outf = open(string("./data/",t.prefix,"/trainProgress.tsv"),"w")

  write(outf,"learning rate:\t$(learningRate)\n")
  write(outf,"momentum rate:\t$(momentumRate)\n")
  write(outf,"epoch\ttrain gold\ttrain correct\ttrain acc.\ttest gold\ttest correct\ttest acc.\tloss\n")

  firstUpdatedFlag = true
  secondUpdatedFlag = true

  for epoch = 1:nepoch
    println("================")
    println("epoch : $(epoch)")
    loss = fit(train_x, train_y, t.model, crossentropy, opt, progress=true)
    println("loss : $(loss)")

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

    println("Train")
    println("\tGold : $(train_total), Correct: $(train_correct)")
    println("\ttest acc.: $(train_correct / train_total)")
    println("Test")
    println("\tGold : $(test_total), Correct: $(test_correct)")
    println("\ttest acc.: $(test_correct / test_total)")
    println("")

    # file output
    write(outf, "$(epoch)\t$(train_total)\t$(train_correct)\t$(train_correct/train_total)\t$(test_total)\t$(test_correct)\t$(test_correct/test_total)\t$(loss)\n")

    epoch % 100 == 0 && flush(outf)
	epoch % (nepoch/10) == 0 && h5save(string("./model/",t.prefix,"/tokenizer_",string(epoch),".h5"),t)
    if (epoch > nepoch * 0.2) && firstUpdatedFlag 
        learningRate *= Float32(0.1)
        firstUpdatedFlag = false
    end
    if epoch > nepoch * 0.6 && secondUpdatedFlag 
        learningRate *= Float32(0.1)
        secondUpdatedFlag = false
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
