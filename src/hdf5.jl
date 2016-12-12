import Merlin: h5save, h5writedict, h5dict, h5convert, h5load!, h5load
export h5convert, h5dict, h5load, h5load!, h5loadTokenizer

h5convert{T}(x::Array{T}) = x

function h5convert(x::Tuple)
    dict = h5dict(Tuple)
    for i = 1:length(x)
        dict[string(i)] = h5convert(x[i])
    end
    dict
end

h5convert(x::Bool) = convert(Int32, x)

# Convolution
function h5convert(f::Merlin.Conv)
    N = length(f.filterdims)
    dict = h5dict(Merlin.Conv{N}, "w"=>f.w, "filterdims"=>f.filterdims, "stride"=>f.stride,"paddims"=>f.paddims)
end

function h5load!{N}(::Type{Merlin.Conv{N}}, data)
    filterdims = h5load!(data["filterdims"])
    paddims = h5load!(data["paddims"])
    stride = h5load!(data["stride"])
    w = h5load!(data["w"])

    # Number of dimension
    N = length(filterdims)
    T = typeof(w.data).parameters[1]

    channeldims = (size(w.data, N+1),size(w.data, N+2))
    conv = Merlin.Conv(T, filterdims, channeldims, stride = stride, paddims = paddims)
    conv.w = w
    conv
end

function h5load!(::Type{Tuple}, data)
    tupl = (data["1"],)
    for i = 2:length(data)
        tupl = tuple(tupl...,(data[string(i)],)...)
    end
    tupl
end

# Loading

h5loadTokenizer(filename::String; outputFile::String="") = h5loadTokenizer!(h5read(filename, "Merlin"), outputFile)

function h5loadTokenizer!(data::Dict,filename::String)
    if haskey(data,"#TYPE") && data["#TYPE"] == "testJukaiNLP.Tokenizer"
        delete!(data,"#TYPE")
        tagset = h5loadTag!(data["tagset"])
        iddict = h5loadId!(data["iddict"])
        model = data["model"]
        delete!(model, "#TYPE")

        T = Float32

		nodeDict = Dict{String, Merlin.Functor}()

        lsIndex = 1

        for (k,v) in model
            typeName = v["1"]["#TYPE"]
            if typeName == "Merlin.Embedding"
                embed = h5load!(v["1"])
				nodeDict["embed"] = embed
            end
            if startswith(typeName,"Merlin.Conv")
                conv = h5load!(v["1"])
				nodeDict["conv"] = conv
            end
            if typeName == "Merlin.Linear"
                ls = h5load!(v["1"])
                nodeDict["ls$(lsIndex)"] = ls
                lsIndex += 1
            end
        end

		x = Var()
		y = Embedding(T, 100000, emboutCh)(x)
		y = Conv(T, (emboutCh, convFilterWidth), (1,convOutCh), paddims=(0,convOutCh))(y)
		y = reshape(y, size(y,2), size(y,3))
		y = transpose(y)
		y = relu(y)
		y = dropout(y, 0.5, true)(y)
		y = Linear(T, convOutCh, lsOutCh)(y)
		y = dropout(y, 0.25, true)(y)
		y = Linear(T, lsOutCh, lsOutCh2)
		g = compile(y, x)
#        g = @graph begin
#          chars = identity(:chars)
#          x = Var(reshape(chars, 1, length(chars)))
#          x = nodeDict["embed"](x)
#          x = nodeDict["conv"](x)
#          x = reshape(x, size(x, 2), size(x, 3))
#          x = transpose(x)
#          x = relu(x)
#		  x = dropout(x, 0.5, false)
#          x = nodeDict["ls1"](x)
#          x = nodeDict["ls2"](x)
#          x
#        end

        t = Tokenizer(filename, iddict, tagset, g)
    else
        t = Tokenizer(filename)
    end
    t
end

#function h5load!(data::Dict, fl::Bool)
#    if fl
#        println("Local h5load")
#    end
#    if haskey(data, "#TYPE")
#        println(data["#TYPE"])
#        T = eval(parse(data["#TYPE"]))
#        delete!(data, "#TYPE")
#        h5load!(T, data)
#    else
#        for (k, v) in data
#            typeof(v) <: Dict && (data[k] = h5load!(v))
#        end
#        data
#    end
#end

#h5load!(::Type{Function}, data) = eval(parse(data["f"]))
#h5load!(::Type{Symbol},data) = parse(data["s"])
#h5load!(::Type{DataType}, data) = eval(parse(data["t"]))
#h5load!(x::Number) = x
#h5load!{T}(x::Array{T}) = x
#h5load!(x::String) = x
#
#function h5load!(::Type{Vector{Any}}, data::Dict)
#    vec = []
#    for (k,v) in data
#        i = parse(Int, k)
#        while i > length(vec)
#            push!(vec, nothing)
#        end
#        vec[i] = h5load!(v)
#    end
#    vec
#end
