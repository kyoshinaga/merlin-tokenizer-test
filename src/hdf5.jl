export h5save, h5writedict, h5dict, h5convert

function h5save(filename::String, data)
    h5open(filename,"w") do h
        h["version"] = string(VERSION)
        g = g_create(h, "testJukaiNLP")
        h5writedict(g, h5convert(data))
    end
end

function h5writedict(g, data::Dict)
    for (k,v) in data
        if typeof(v) <: Dict
            c = g_create(g, string(k))
            h5writedict(c, v)
        else
            g[string(k)] = v
        end
    end
end

function h5dict(T::Type, x::Pair...)
    dict = Dict{String, Any}("#TYPE" => string(T))
    for (k,v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

h5convert(x::Number) = x
h5convert{T}(x::Array{T}) = x
h5convert(x::String) = x
h5convert(x::Symbol) = h5dict(Symbol, "s"=>string(x))
h5convert(x::Function) = h5dict(Function, "f"=>string(x))
h5convert(x::DataType) = h5dict(DataType, "t"=>string(x))

function h5convert(x::Dict)
    dict = Dict{String, Any}()
    for (k, v) in x
        dict[string(k)] = h5convert(v)
    end
    dict
end

function h5convert(x::Vector{Any})
    dict = h5dict(Vector{Any})
    for i = 1:length(x)
        dict[string(i)] = h5convert(x[i])
    end
    dict
end
