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
      if line == "EOS"
          length(sent) > 0 && push!(doc, sent)
          sent = []
          newflag = true
      else
          items = split(line, ' ')
          if index == 1
              items = [items[1], '_']
          elseif newflag
              items = [items[1], 'N']
              newflag = false
          elseif items[4] == "特殊"
              #if items[1] == "、"
              #    items = [items[1], '_']
              #else
                  items = [items[1], 'S']
              #end
          else
              items = [items[1], '_']
          end
          push!(sent, items)
      end
  end
  length(sent) > 0 && push!(doc, sent)
  doc
end

function readBCCWJ(path)
    xdoc = parse_file(path)
    xroot = LightXML.root(xdoc)
    doc = []
    flattenSUW!(xroot, doc)
    doc
end

function flattenSUW!{T<:AbstractXMLNode}(r::T, v::Vector)
    if name(r) != "SUW"
        for c in child_nodes(r)
            if name(c) == "sentence"
                sent = []
                flattenSUW!(c, sent)
                sent[1][2] = "N"
                push!(v, sent)
            else
                flattenSUW!(c, v)
            end
        end
    else
        elem = XMLElement(r)
        text = content(elem)
        atr = (attribute(elem, "wType") == "記号" ? "S" : "_")
        push!(v, [text, atr])
    end
end

function flattenLUW!{T<:AbstractXMLNode}(r::T, v::Vector)

end
