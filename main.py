import os
import subprocess
import yaafelib as yl

def writeTrainFeatures(dataPath, featPath):
    
    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    instrList = [item for item in os.listdir(dataPath) if not item.endswith('.txt')]
    instrIndex = dict(zip(instrList, range(len(instrList))))
    limit = 100 # The number of audio files for each instrument that's used

    for instr in instrList:

        count = 0
        subStr = '[' + instr + ']' + '[nod]'

        for audioFile in (fn for fn in os.listdir(os.path.join(dataPath, instr)) if subStr in fn):

            count += 1

            if (count < limit):

                # Get features
                afp.processFile(eng, os.path.join(dataPath, instr, audioFile))
                feats = eng.readAllOutputs()
                featList = list(feats.keys())

                # Update the number of frames
                numFrames = len(feats.get(featList[0]))
                totalFrames += numFrames

                # Write features
                for frInd in range(numFrames):
                    for feat in featList:
                        for val in feats.get(feat)[frInd]:
                            featFile.write('%.8f ' % val)
                    
                    featFile.write('%d' % instrIndex.get(instr))
                    featFile.write('\n')
    
    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()

    return instrIndex


def writeTestFeatures(dataPath, featPath, instrIndex):

    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    
    for audioFile in (fn for fn in os.listdir(dataPath) if fn.endswith('.wav')):
        
        # Get the label
        subStr = audioFile.split('.')
        filename = ".".join(subStr[0 : len(subStr) - 1])
        labelFile = open(os.path.join(dataPath, filename + '.txt'), 'r')
        label = labelFile.readline().strip()
        otherInstr = labelFile.readline()
        
        if (not otherInstr.isspace() and label in instrIndex):
            
            # Get features
            afp.processFile(eng, os.path.join(dataPath, audioFile))
            feats = eng.readAllOutputs()
            featList = list(feats.keys())

            # Update the number of frames
            numFrames = len(feats.get(featList[0]))
            totalFrames += numFrames

            # Write features
            for frInd in range(numFrames):
                for feat in featList:
                    for val in feats.get(feat)[frInd]:
                        featFile.write('%.8f ' % val)
                
                featFile.write('%d' % instrIndex.get(label))
                featFile.write('\n')

    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()


# Main

# trainAudio = '/home/deniz/Documents/IRMAS/IRMAS-Sample/Training/'
trainAudio = '/home/deniz/Documents/IRMAS/BinaryTrainingData/'
trainFeats = './binaryModel/trainFeatures.dat'
testAudio = '/home/deniz/Documents/IRMAS/TestingData/'
testFeats = './binaryModel/testFeatures.dat'
model = './binaryModel/model.svm'

# Specify features
fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512 CepsNbCoeffs=13 FFTWindow=Hamming')
fp.addFeature('lsf: LSF blockSize=1024 stepSize=512 LSFNbCoeffs=7')
# Dimensions: mfcc +13, lsf +3, sss +4, tss +4
dimensions = 20 # The sum of the dimensions of the features

# Initialize yaafe tools
df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

# Write training features
print('\nExtracting training features\n')
instruments = writeTrainFeatures(trainAudio, trainFeats)

# Train the svm 
print('\nTraining SVM\n')
subprocess.call(['./trainSVM', '-multi', trainFeats, model])

# Write cross-validation features
print('\nExtracting testing features\n')
writeTestFeatures(testAudio, testFeats, instruments)

# Cross-validate the model
print('\nTesting SVM\n')
subprocess.call(['./testSVM', '-multi', model, testFeats])
