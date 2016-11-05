export Tokenizer, TokenizerAutoEncode, h5convert

type Tokenizer <: Functor
  prefix::String
  dict::IdDict
  tagset::Tagset
  model
end

function Tokenizer(prefix::String)
  dict = IdDict(map(String, ["UNKNOWN", " ","\n"]))
  T = Float32
  emboutCh = 12
  convFilterWidth = 7
  convOutCh = 128
  convPadWidth = Int((convFilterWidth - 1)/2)
  lsOutCh = 3
  embed = Embedding(T, 10000, emboutCh)
  conv = Conv(T, (emboutCh,convFilterWidth),(1,convOutCh),paddims=(0,convPadWidth))
  ls = Linear(T, convOutCh, lsOutCh)
  g = @graph begin
    chars = identity(:chars)
    x = Var(reshape(chars, 1, length(chars)))
    x = embed(x)
    x = conv(x)
    x = reshape(x, size(x, 2), size(x, 3))
    x = transpose(x)
    x = relu(x)
#	x = dropout(x, 0.5, true)
    x = ls(x)
    x
  end
  outf = open(string("./data/",prefix,"/NetworkConstruction.tsv"),"w")
  write(outf, "Embediding: ($(length(embed.ws)),$(length(embed.ws[1].data)))\n")
  write(outf, "Conv:\n")
  write(outf, "\tfilterdims: $(conv.filterdims)\n")
  write(outf, "\tch: ($(size(conv.w.data,3)),$(size(conv.w.data,4)))\n")
  write(outf, "\tstride: $(conv.stride)\n")
  write(outf, "\tpaddims: $(conv.paddims)\n")
  write(outf, "Linear: ($(size(ls.w, 2)),$(size(ls.w, 1)))\n")
  close(outf)

  Tokenizer(prefix, dict, IOE(), g)
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

Base.size(x:: Var, dims:: Int) = size(x.data, dims)
