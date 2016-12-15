using Formatting

function train(t::Tagger, nepoch::Int, train_x, train_y, test_x, test_y)
	opt = SGD()

	for epoch = 1:nepoch
		println("epoch: $(epoch)")
		opt.rate = 0.0075 / epoch
		loss = fit(train_x, train_y, t.model, crossentropy, opt)
		println("Train loss: $(loss)")

		test_z = map(x -> predict(t.model, x), test_x)
		acc = accuracy(test_y, test_z)
		println("Valid acc.: $(acc)")
		println("")
	end
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
