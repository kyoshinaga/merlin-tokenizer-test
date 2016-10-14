using LightXML
importall LightXML
export readconll,readknp, readBCCWJ

function readconll(path, columns=Int[])
  doc = []
  sent = []
  lines = open(readlines, path)
  for line in lines
    line = chomp(line)
    if length(line) == 0
      length(sent) > 0 && push!(doc, sent)
      sent = []
    else
      items = split(line, '\t')
      length(items) > 0 && (items = items[columns])
      push!(sent, items)
    end
  end
  length(sent) > 0 && push!(doc, sent)
  doc
end

function readknp(path)
  doc = []
  sent = []
  lines = open(readlines,path)
  comment = Char['*','#']
  newflag = false
  index = 0
  for line in lines
      if startswith(line, comment)
          continue
      end
      index += 1
      line = chomp(line)
      tag = ""
      if line == "EOS"
          length(sent) > 0 && push!(doc, sent)
          sent = []
          newflag = true
      else
          items = split(line, ' ')
		  tag = (items[4] != "特殊" ? "_" : "S")
          if index != 1 && newflag
			  tag = string(tag, "N")
			  newflag = false
		  end
		  word = [items[1], tag]
          push!(sent, word)
      end
  end
  length(sent) > 0 && push!(doc, sent)
  doc
end

function readBCCWJ(path)
    xdoc = parse_file(path)
    xroot = LightXML.root(xdoc)
    suwDoc = []
    flattenSUW!(xroot, suwDoc)
    luwDoc = []
    flattenLUW!(xroot, luwDoc, "")
    suwDoc, luwDoc
end

function flattenSUW!{T<:AbstractXMLNode}(r::T, v::Vector)
    if name(r) != "SUW"
        for c in child_nodes(r)
            if name(c) == "sentence"
                sent = []
                flattenSUW!(c, sent)
                length(sent) > 0 && (sent[1][2] = string(sent[1][2],"N"))
                push!(v, sent)
            else
                flattenSUW!(c, v)
            end
        end
    else
        text = getText(r)
        atr = (getAttribute(r, "wType") == "記号" ? "S" : "_")
        push!(v, [text, atr])
    end
end

"""
	flattenLUW!(r, v; lowPos)

Adding LUW tag to BCCWJ corpus.
Tag definition (Ref. K. Uchimoto et al., 2007):
	Ba : Beginning of word. POS information agrees.
	Ia : Middle or End. POS information agrees.
	B  : Beginning of word. POS information does not agree.
	I  : Middle or End. POS information does not agree.
"""
function flattenLUW!{T<:AbstractXMLNode}(r::T, v::Vector, luwPos::String)
    if name(r) != "SUW"
        for c in child_nodes(r)
            if name(c) == "sentence"
                sent = []
                flattenLUW!(c, sent, "")
                length(sent) > 0 && (sent[1][2] = string(sent[1][2],"N"))
                push!(v, sent)
            elseif name(c) == "LUW"
                pos = getPos(c, true)
                flattenLUW!(c, v, string(pos))
            else
                flattenLUW!(c, v, luwPos)
            end
        end
    else
        text = getText(r)
        pos = getPos(r, false)
		atr = (luwPos == pos ? "a" : "")
        push!(v, [text, atr])
    end
end

"""
    getPos(n, luwFlag)

Return refined pos information.

* n: Node
* luwFlag: Flag of node type. True => LUW, False = SUW.
"""
function getPos{T<:AbstractXMLNode}(n::T, luwFlag::Bool)
    pos = getAttribute(n, (luwFlag ? "l_pos" : "pos"))
    posList = split(pos,"-")
    pos = (length(posList) > 1) ? join(posList[1:2],"-") : pos
    pos
end

getAttribute{T<:AbstractXMLNode}(n::T, str::String) = attribute(XMLElement(n), str)

getText{T<:AbstractXMLNode}(n::T) = content(XMLElement(n))
