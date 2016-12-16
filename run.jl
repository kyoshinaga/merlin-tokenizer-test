include("./src/bioTagging.jl")
using bioTagging.tagging

using Merlin
using CPUTime

import Merlin:save

function doTest(trainData, validData, prefix::String, nepoch::Int,wordEmbCh, wordWindow, charEmbCh, charWindow)

    success(`mkdir -p ./data/$(prefix)`)
    success(`mkdir -p ./model/$(prefix)`)

    t, train_x, train_y, valid_x, valid_y = Tagger(trainData, validData, prefix=prefix, 
                                                   wordEmbDim=wordEmbCh, wordWindow=wordWindow, 
                                                   charEmbDim=charEmbCh, charWindow=charWindow)

    beginTime = time()
    @time @CPUtime train(t, nepoch, train_x, train_y, valid_x, valid_y)
    endTime = time()

    outf = open("./data/$(prefix)/computeTime.txt","w")
    write(outf,"time:\t$(endTime - beginTime)")
    close(outf)

    save(string("./model/",prefix,"/tagger_result.h5"),"w","Merlin", t)
    t
end


jpnTrainDoc = "./corpus/jpnTrainDoc.h5"
jpnValidDoc = "./corpus/jpnValidDoc.h5"
#jpnTrainDoc = "./corpus/sampleDoc.h5"
#jpnValidDoc = "./corpus/sampleDoc.h5"

nepoch = 100

# pattern1
prefix = "20161216/pattern1"
wordEmbCh = 64
wordWindow = 3
charEmbCh = 12
charWindow = 5

t = doTest(jpnTrainDoc,jpnValidDoc, prefix, nepoch, wordEmbCh, wordWindow, charEmbCh, charWindow)

# pattern2
prefix = "20161216/pattern2"
wordEmbCh = 64
wordWindow = 5
charEmbCh = 12
charWindow = 5

t = doTest(jpnTrainDoc,jpnValidDoc, prefix, nepoch, wordEmbCh, wordWindow, charEmbCh, charWindow)

# pattern3
prefix = "20161216/pattern3"
wordEmbCh = 64
wordWindow = 7
charEmbCh = 12
charWindow = 5

t = doTest(jpnTrainDoc,jpnValidDoc, prefix, nepoch, wordEmbCh, wordWindow, charEmbCh, charWindow)

# pattern4
prefix = "20161216/pattern4"
wordEmbCh = 64
wordWindow = 9
charEmbCh = 12
charWindow = 5

t = doTest(jpnTrainDoc,jpnValidDoc, prefix, nepoch, wordEmbCh, wordWindow, charEmbCh, charWindow)
