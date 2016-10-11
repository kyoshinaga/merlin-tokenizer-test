export encode, convJukai, flatten, flattenDoc, train

function encode(t::Tokenizer, doc::Vector)
  unk, space, lf = t.dict["UNKNOWN"], t.dict[" "], t.dict["\n"]
  chars = Int[]
  ranges = UnitRange{Int}[]
  pos = 1
  for sent in doc
    for (word, tag) in sent
      for c in tag
        # c == '_' && continue
        # c == 'S' && continue
        # if c == 'S' # Space
          # push!(chars, space)
        #elseif c == 'N' # Newline
        if c == 'N'
          push!(chars, lf)
          pos += 1
        end
      end
      for c in word
        push!(chars, push!(t.dict, string(c)))
      end
      tag != "S" && push!(ranges, pos:pos+length(word) - 1)
      pos += length(word)
    end
  end
  chars, ranges
end

function train(t::Tokenizer, nepoch::Int, trainData::Vector, testData::Vector)
  chars, ranges = encode(t, trainData)
  tags = encode(t.tagset, ranges, length(chars))
  train_x, train_y = [], []
  push!(train_x, chars)
  push!(train_y, tags)

  chars2, ranges2 = encode(t, testData)
  tags2 = encode(t.tagset, ranges2, length(chars2))
  test_x, test_y = [], []
  push!(test_x, chars2)
  push!(test_y, tags2)

  opt = SGD(0.000001, momentum=0.9)

  outf = open(t.filename,"w")

  write(outf,"epoch\ttrain gold\ttrain correct\ttrain acc.\ttest gold\ttest correct\ttest acc.\tloss\n")

  for epoch = 1:nepoch
    println("================")
    println("epoch : $(epoch)")
    #loss = fit(t.model, crossentropy, opt, train_x, train_y)
    loss = fit(train_x, train_y, t.model, crossentropy, opt, progress=true)
    println("loss : $(loss)")

    train_z = map(train_x) do x
        argmax(t.model(x).data, 1)
    end

    test_z = map(test_x) do x
      argmax(t.model(x).data, 1)
    end

    train_correct, train_total = accuracy(flatten(train_y), flatten(train_z))
    correct, total = accuracy(flatten(test_y), flatten(test_z))

    println("Train")
    println("\tGold : $(train_total), Correct: $(train_correct)")
    println("\ttest acc.: $(train_correct / train_total)")
    println("Test")
    println("\tGold : $(total), Correct: $(correct)")
    println("\ttest acc.: $(correct / total)")
    println("")

    # file output
    write(outf, "$(epoch)\t$(train_total)\t$(train_correct)\t$(train_correct/train_total)\t$(total)\t$(correct)\t$(correct/total)\t$(loss)\n")

    epoch % 100 == 0 && flush(outf)

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
