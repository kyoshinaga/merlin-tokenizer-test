abstract Tagset

export encode, decode, h5convert, h5load, IOE, BI

immutable IOE <: Tagset
	I::Int
	O::Int
	E::Int
end

IOE() = IOE(1,2,3)

function decode(tagset::IOE, tags::Vector{Int})
	bpos = 0
	ranges = UnitRange{Int}[]
	for i = 1:length(tags)
		t = tags[i]
		t != tagset.O && bpos == 0 && (bpos = i)
		if t == tagset.E
			push!(ranges, bpos:i)
			bpos = 0
		elseif t == tagset.O
			push!(ranges, i:i)
		end
	end
	ranges
end

function encode(tagset::IOE, ranges::Vector{UnitRange{Int}}, length::Int)
	# tags = fill(tagset.O, last(ranges[end]))
	tags = fill(tagset.O, length)
	for r in ranges
		tags[r] = tagset.I
		tags[last(r)] = tagset.E
	end
	tags
end

function h5convert(f::IOE)
	h5dict(IOE, "I"=>f.I, "O"=>f.O, "E"=>f.E)
end

h5loadTag!(data) = IOE(data["I"],data["O"],data["E"])

immutable BI <: Tagset
	Ba::Int
	Ia::Int
	B::Int
	I::Int
end

BI() = BI(1,2,3,4)

function decode(tagset::BI, tags::Vector{Int})
	bpos = 0
	ranges = UnitRange{Int}[]
	for i = 1:length(tags)
		t = tags[i]
		t != tagset.O && bpos == 0 && (bpos = i)
		if t == tagset.E
			push!(ranges, bpos:i)
			bpos = 0
		end
	end
	ranges
end

function encode(tagset::BI, ranges::Vector{UnitRange{Int}}, length::Int)
	# tags = fill(tagset.O, last(ranges[end]))
	tags = fill(tagset.Ba, length)
	for r in ranges
		tags[r] = tagset.I
		tags[last(r)] = tagset.E
	end
	tags
end

function h5convert(f::BI)
	h5dict(IOE, "Ba"=>f.Ba, "Ia"=>f.Ia, "B"=>f.B, "I"=>f.I)
end

h5loadTagBI!(data) = BI(data["Ba"],data["Ia"],data["B"],data["I"])
