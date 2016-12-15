type Model
    wordfun
    charfun
    sentfun
end

function Model(wordembeds, charembeds, ntags::Int,
	charEmbDim, charWindow,
	wordEmbDim, wordWindow)

    T = Float32

	# Character part
	charWindowWidth = charWindow * charEmbDim
	charPaddimg = Int((charWindow - 1) / 2) * charEmbDim
    x = Var()
    y = charembeds(x)
    y = window(y, (charWinowWidth,), strides=(charEmbDim,), pads=(charPadding,))
    y = Linear(T, charWinowWidth, charWinowWidth)(y)
    y = max(y, 2)
    charfun = compile(y, x)

	# Word part
	concatWordUnit = wordEmbDim + charWinowWidth
	wordWindowWidth = concatWordUnit * wordWindow
	wordPadding = Int((wordWindow - 1) / 2) * concatWordUnit
	
    w = Var() # word vector
    c = Var() # chars vector
    y = wordembeds(w)
    y = concat(1, y, c)
    y = window(y, (wordWindowWidth,), strides=(concatWordUnit,), pads=(wordPadding,))
    y = Linear(T,wordWindowWidth,wordPadding)(y)
    y = relu(y)
    y = Linear(T, wordPadding, ntags)(y)
    sentfun = compile(y, w, c)

    Model(wordembeds, charfun, sentfun)
end

function (m::Model)(tokens::Vector{Token})
    wordvec = map(t -> t.word, tokens)
    wordvec = reshape(wordvec, 1, length(wordvec))
    charvecs = map(tokens) do t
        charvec = reshape(t.chars, 1, length(t.chars))
        m.charfun(Var(charvec))
    end
    charmat = concat(2, charvecs)
    m.sentfun(Var(wordvec), charmat)
end
