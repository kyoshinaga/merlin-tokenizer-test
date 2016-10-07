export Tokenizer, TokenizerAutoEncode, h5convert

type Tokenizer <: Functor
  dict::IdDict
  tagset::Tagset
  model
end

function Tokenizer()
  dict = IdDict(map(String, ["UNKNOWN", " ","\n"]))
  T = Float32
  embed = Embedding(T, 100000, 10)
  conv = Conv(T, (10,9),(1,100),paddims=(0,4))
  # conv = Conv(T, (10,9),(1,128),paddims=(0,4))
  ls = Linear(T, 100, 4)
  # ls = Linear(T, 128, 4)
  g = @graph begin
    chars = identity(:chars)
    x = Var(reshape(chars, 1, length(chars)))
    x = embed(x)
    x = conv(x)
    x = reshape(x, size(x, 2), size(x, 3))
    x = transpose(x)
    x = relu(x)
    x = ls(x)
    x
  end
  Tokenizer(dict, IOE(), g)
end

function TokenizerAutoEncode()
  dict = IdDict(map(String, ["UNKNOWN", " ","\n"]))
  T = Float32
  embed = Embedding(T, 100000, 10)
  conv = Conv(T, (10,9),(1,100),paddims=(0,4))
  ls = [Linear(T, 100, 4),Linear(T, 4, 100000)]
  g = @graph begin
    chars = identity(:chars)
    x = Var(reshape(chars, 1, length(chars)))
    x = embed(x)
    x = conv(x)
    x = reshape(x, size(x, 2), size(x, 3))
    x = transpose(x)
    x = relu(x)
    x = ls[1](x)
    x = ls[2](x)
    x
  end
  Tokenizer(dict, IOE(), g)
end

function TokenizerCuda()
  dict = IdDict(map(String, ["UNKNOWN", " ","\n"]))
  T = Float32
  embed = Embedding(T, 100000, 10)
  conv = Conv(T, (10,9),(1,100),paddims=(0,4))
  ls = Linear(T, 100, 4)
  g = @graph begin
    chars = identity(:chars)
    x = Var(reshape(chars, 1, length(chars)))
    x = embed(x)
    x = conv(x)
    x = reshape(x, size(x, 2), size(x, 3))
    x = transpose(x)
    x = VarToCuArray(x)
    x = relu(x)
    x = ls(x)
    x = CuArrayToVar(x)
    x
  end
  Tokenizer(dict, IOE(), g)
end

function VarToCuArray(x:: Var)
  y = x
  y.data = CuArray(x.data)
  y
end

function CuArrayToVar(x:: Var)
  y = x
  y.data = Array(x.data)
  y
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

function h5convert(f::Tokenizer)
    h5dict(Tokenizer, "tagset"=>f.tagset, "iddict"=>f.dict, "model"=>f.model)
    # h5dict(Tokenizer, "tags"=>f.tagset, "iddict"=>f.dict)
end

function h5load!(::Type{Tokenizer}, data)

    tagset = h5load!(data["tagset"])
    iddict = h5load!(data["iddict"])
    model = h5load!(data["model"])

    Tokenizer(iddict, tagset, model)
end
