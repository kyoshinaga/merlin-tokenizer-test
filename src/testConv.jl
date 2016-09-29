function convTest{T}(w::Array{T}, x::Array{T}, windims, stride, paddims)
    N = length(windims)
    outdims = outsize(x, windims, stride, paddims)
    work = Array(T, prod(outdims),prod(windims) *size(x,N+1), size(x,N+2))
    println(string("outdims: ",outdims))
    println(string("work size: ", size(work)))
    returnWork = im2col(x, work, windims, stride, paddims)

    w = reshape(w, size(work, 2), size(w, N+2))

    y = returnWork[:,:,1] * w

    y = reshape(y, outdims..., size(y,2), size(y,3))

    w, returnWork, y
end

function im2col{T}(x::Array{T}, y:: Array{T}, windims, stride, paddims)
    N = length(windims)
    xsize = Cint[size(x,i) for i=1:N+1]
    xsize[N+1] *= size(x, N+2)

    println(string("xsize: ", xsize))
    println(string("ysize: ", size(y)))

    x1 = xsize[1] # 10
    x2 = xsize[2] # 20
    x3 = xsize[3] # 1

    w1 = windims[1] # 10
    w2 = windims[2] # 9

    s1 = stride[1] # 1
    s2 = stride[2] # 1

    p1 = paddims[1] # 0
    p2 = paddims[2] # 4

    n1 = Int((x1 + 2 * p1 - w1) / s1 + 1) # 1
    n2 = Int((x2 + 2 * p2 - w2) / s2 + 1) # 20

    o = Int(1)

    for d3 = 0:(x3 - 1)                 # 0 -> 0
        for d2 = 0:(w2 - 1)             # 0 -> 8
            for d1 = 0:(w1 - 1)         # 0 -> 9
                for k2 = 0:(n2 - 1)     # 0 -> 19
                    for k1 = 0:(n1 - 1) # 0 -> 0
                        i1 = Int(k1 * s1 - p1 + d1)  # initial 0 * 1 - 0 + 0 + 1 final 0 * 1 - 0 + 9 + 1
                        i2 = Int(k2 * s2 - p2 + d2)  # initial 0 * 1 - 4 + 0 + 1 final 19 * 1 - 4 + 8 + 1
                        if i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2
                            i = Int(i1 + x1 * (i2 + x2 * d3) + 1) 
                            y[o] = x[i]
                        else
                            y[o] = 0
                        end
                        o += 1
                    end
                end
            end
        end
    end
    println(string("Finish index: ",o))
    y
end

function outsize(x::Array, windims, stride, paddims)
    N = length(windims)
    Int[(size(x,i) + 2 * paddims[i] - windims[i]) / stride[i] + 1 for i = 1:N]
end
