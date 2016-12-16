using Formatting

function train(t::Tagger, nepoch::Int, train_x, train_y, valid_x, valid_y)
	opt = SGD()

    outProgress = open("./data/$(t.prefix)/trainProgress.tsv","w")
    write(outProgress,"epoch\ttrain acc.\tvalid acc.\ttrain loss\tvalid loss\n")
    fmt = "%1.5f"

	for epoch = 1:nepoch
        println("===============")
		println("epoch: $(epoch)")
		opt.rate = 0.0075 / epoch
		trainLoss = fit(train_x, train_y, t.model, crossentropy, opt)
        train_z = map(x -> predict(t.model, x), train_x)
        trainAcc = accuracy(train_y, train_z)

        trainLoss = sprintf1(fmt, trainLoss)
        trainAcc = sprintf1(fmt, trainAcc)

		valid_z = map(x -> predict(t.model, x), valid_x)
		validAcc = accuracy(valid_y, valid_z)
        validLoss = 0
        for data in zip(valid_x, valid_y)
            z = t.model(data[1])
            l = crossentropy(data[2],z)
            validLoss += sum(l.data)
        end

        validLoss = sprintf1(fmt, validLoss)
        validAcc = sprintf1(fmt, validAcc)

        println("")
		println("Train loss: $(trainLoss)")
        println("Train acc: $(trainAcc)")
        println("")
        println("Valid loss: $(validLoss)")
		println("Valid acc.: $(validAcc)")
		println("")

        write(outProgress, "$(epoch)\t$(trainAcc)\t$(validAcc)\t$(trainLoss)\t$(validLoss)\n")

        if epoch % (nepoch / 10) == 0
            save("./model/$(t.prefix)/tagger_$(epoch).h5","w","Merlin",t)
            flush(outProgress)
        end
        
	end

    close(outProgress)
end

predict(model, data) = argmax(model(data).data, 1)

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
