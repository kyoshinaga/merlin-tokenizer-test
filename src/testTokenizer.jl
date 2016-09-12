type Tokenizer
  dict::IdDict
  tagset::Tagset
  model
end

function Tokenizer()
  dict = IdDict(map(UTF8String, ["UNKNOWN", " ","\n"]))
  T = Float32
  embed = Embedding(T, 10000, 10)
  conv = Conv(T, (10,7),(1,70),paddims=(0,3))
  ls = Linear(T, 70, 4)
  g = @graph begin
    chars = identity(:chars)
    x = Var(reshape(chars, 1, length(chars)))
    x = embed(x)
    x = conv(x)
    x = reshape(x, sizeVar(x, 2), sizeVar(x, 3))
    x = transpose(x)
    x = relu(x)
    x = ls(x)
    x
  end
  Tokenizer(dict, IOE(), g)
end

function sizeVar(x:: Var, dim:: Int)
  size(x.data, dim)
end

function (t::Tokenizer)(chars::Vector{Char})
  unk = t.dict["UNKNOWN"]
  x = map(chars) do c
    get(t.dict, string(c), unk)
  end
  y = t.model(x).data
  tags = argmax(y, 1)
  decode(t.tagset, tags)
end

(t::Tokenizer)(str:: String) = t(Vector{Char}(str))
