function getAll(x:: SubString)
    totalNumOfFiles = 0
    prefix = "./data"
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

prefix = "./data"
dirList = readstring(`ls $(prefix)`)
dirList = split(chomp(dirList),'\n')
numList = length(dirList)
doneList = String[]

totalNumOfFiles = 0

totalNumOfFiles += map(getAll, dirList)
println(sum(totalNumOfFiles,1))
