function getAll(x:: SubString)
    totalNumOfFiles = 0
    prefix = "./yahoo"
    success(`ls ./corpus/$(x)`) || run(`mkdir ./corpus/$(x)`)
    fileList = []
    fileList = readstring(`ls $(prefix)/$(x)`)
    fileList = split(chomp(fileList), '\n')
    numFile = length(fileList)
    totalNumOfFiles += numFile
    map(fileList) do file
        jumanResult = readstring(pipeline(`cat ./$(prefix)/$(x)/$(file)`,`juman`))
        outf = open("./corpus/$(x)/$(file).juman", "w")
        write(outf, jumanResult)
        close(outf)
    end
    println("$(x), Num of file: $(totalNumOfFiles)")
    totalNumOfFiles
end

prefix = "./yahoo"
dirList = readstring(`ls $(prefix)`)
dirList = split(chomp(dirList),'\n')
numList = length(dirList)
doneList = String[]

totalNumOfFiles = 0

totalNumOfFiles += map(getAll, dirList)
println(sum(totalNumOfFiles,1))

prefix = "./corpus"
dirList = readstring(`ls $(prefix)`)
dirList = split(chomp(dirList),'\n')
numList = length(dirList)
doneList = String[]

doc = []

map(dirList) do dir
    fileList = []
    fileList = readstring(`ls $(prefix)/$(dir)`)
    fileList = split(chomp(fileList), '\n')
    numFile = length(fileList)
    map(fileList) do file
        push!(doc, readJuman("./$(prefix)/$(dir)/$(file)"))
    end
end

doc = flattenDoc(doc)

numOfData = length(doc)
numOfTrainData = Int(floor(0.8 * numOfData))
numOfValidData = Int(floor(0.9 * numOfData))
pickItemList = randperm(numOfData)
jpnTrainDoc = copy(doc[pickItemList[1:numOfTrainData]])
jpnValidDoc = copy(doc[pickItemList[(numOfTrainData+1):numOfValidData]])
jpnTestDoc = copy(doc[pickItemList[(numOfValidData+1):numOfData]])

println("train length: $(length(jpnTrainDoc))")
println("train length: $(length(jpnValidDoc))")
println("train length: $(length(jpnTestDoc))")

h5save("./corpus/jpnTrainDoc.h5",jpnTrainDoc)
h5save("./corpus/jpnValidDoc.h5",jpnValidDoc)
h5save("./corpus/jpnTestDoc.h5",jpnTestDoc)
