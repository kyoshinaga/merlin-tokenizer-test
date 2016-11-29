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
	I_Ba::Int # 1
	I_Ia::Int # 2
	I_B::Int  # 3
	I_I::Int  # 4
	O_Ba::Int # 5
	O_Ia::Int # 6
	O_B::Int  # 7
	O_I::Int  # 8
	E_Ba::Int # 9
	E_Ia::Int # 10
	E_B::Int  # 11
	E_I::Int  # 12
end

BI() = BI(1,2,3,4,5,6,7,8,9,10,11,12)

function decode(tagset::BI, tags::Vector{Int})
	bpos = 0
	ranges = UnitRange{Int}[]
	luwBpos = []
	luwRanges = UnitRange{Int}[]
	lpos = 0
	tagIOE = IOE()
	tagBI = BI()
	for i = 1:length(tags)
		t = tags[i]
		ioe = convert(Int, floor((t - 1) / 4)) + 1
		bi =  (t - 1) % 4 + 1
		ioe != tagIOE.O && bpos == 0 && (bpos = i)
		if ioe == tagIOE.E
			push!(ranges, bpos:i)
			bpos = 0
			lpos += 1
			(bi % 2) == 1 && push!(luwBpos, lpos)
		end
	end
	push!(luwBpos, length(ranges) + 1)
	for i = 2:length(luwBpos)
		push!(luwRanges, luwBpos[i-1]:(luwBpos[i] - 1))
	end
	ranges, luwRanges
end

#function encode(tagset::BI, ranges::Vector{UnitRange{Int}}, length::Int)
#	# tags = fill(tagset.O, last(ranges[end]))
#	tags = fill(tagset.Ba, length)
#	for r in ranges
#		tags[r] = tagset.I
#		tags[last(r)] = tagset.E
#	end
#	tags
#end

function h5convert(f::BI)
	h5dict(IOE,
	"I_Ba"=>f.I_Ba,
	"I_Ia"=>f.I_Ia,
	"I_B"=> f.I_B,
	"I_I"=> f.I_I,
	"O_Ba"=> f.O_Ba,
	"O_Ia"=> f.O_Ia,
	"O_B"=> f.O_B,
	"O_I"=> f.O_I,
	"E_Ba"=>f.E_Ba,
	"E_Ia"=>f.E_Ia,
	"E_B"=> f.E_B,
	"E_I"=> f.E_I
	)
end

h5loadTagBI!(data) = BI(
data["I_Ba"],
data["I_Ia"],
data["I_B"],
data["E_I"],
data["O_Ba"],
data["O_Ia"],
data["O_B"],
data["O_I"],
data["I_I"],
data["E_Ba"],
data["E_Ia"],
data["E_B"]
)
