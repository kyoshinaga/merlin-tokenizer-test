h5convert{T}(x::Array{T}) = x

function h5convert(x::Tuple) 
    dict = h5dict(Tuple)
    for i = 1:length(x)
        dict[string(i)] = h5convert(x[i])
    end
    dict
end

# Convolution
function h5convert(f::Merlin.Conv)
    dict = h5dict(Merlin.Conv, "w"=>f.w, "filterdims"=>f.filterdims, "stride"=>f.stride,"paddims"=>f.paddims)
end
