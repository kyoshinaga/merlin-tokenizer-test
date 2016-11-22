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
	I_Ba::Int
	I_Ia::Int
	I_B::Int
	I_I::Int
	E_Ba::Int
	E_Ia::Int
	E_B::Int
	E_I::Int
	O_Ba::Int
	O_Ia::Int
	O_B::Int
	O_I::Int
end

BI() = BI(1,2,3,4,5,6,7,8,9,10,11,12)

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
    h5dict(IOE,
	"I_Ba"=>f.I_Ba,
	"I_Ia"=>f.I_Ia,
	"I_B"=> f.I_B,
	"I_I"=> f.I_I,
	"E_Ba"=>f.E_Ba,
	"E_Ia"=>f.E_Ia,
	"E_B"=> f.E_B,
	"E_I"=> f.E_I,
	"O_Ba"=> f.O_Ba,
	"O_Ia"=> f.O_Ia,
	"O_B"=> f.O_B,
	"O_I"=> f.O_I
	)
end

h5loadTagBI!(data) = BI(
data["I_Ba"],
data["I_Ia"],
data["I_B"],
data["I_I"],
data["E_Ba"],
data["E_Ia"],
data["E_B"],
data["E_I"],
data["O_Ba"],
data["O_Ia"],
data["O_B"],
data["O_I"]
)
