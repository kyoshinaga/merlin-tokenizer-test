export readconll,readknp

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
        items = [items[1], 'S']
      else
        items = [items[1], '_']
      end
      push!(sent, items)
    end
  end
  length(sent) > 0 && push!(doc, sent)
  doc
end
