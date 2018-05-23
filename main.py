import os
import subprocess
import yaafelib as yl

def writeTrainFeatures(dataPath, featPath):

    featFile = open(featPath, 'w')
    featFile.write('1234578')   # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    dimensions = 18 # The sum of the dimensions of the features
    instrList = os.listdir(dataPath)
    instrIndex = dict(zip(instrList, range(len(instrList))))

    for instr in instrList:
        for audioFile in os.listdir(os.path.join(dataPath, instr)):
            
            # Get features
            afp.processFile(eng, os.path.join(dataPath, instr, audioFile))
            feats = eng.readAllOutputs()
            keys = list(feats.keys())

            # Update the number of frames
            numFrames = len(feats.get(keys[0]))
            totalFrames += numFrames

            # Write features
            for frInd in range(numFrames):
                for feat in keys:
                    for val in feats.get(feat)[frInd]:
                        featFile.write('%.8f ' % val)
                
                featFile.write('%d' % instrIndex.get(instr))
                featFile.write('\n')

    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()

    return instrIndex


def writeTestData(dataPath, featPath, instrIndex):

    featFile = open(featPath, 'w')
    featFile.write('1234578')   # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    dimensions = 18 # The sum of the dimensions of the features
    
    for audioFile in (file for file in  os.listdir(dataPath) if file.endswith('.wav')):
        
        # Get the label
        substr = audioFile.split('.')
        filename = ".".join(substr[0 : len(substr) - 1])
        labelFile = open(os.path.join(dataPath, filename + '.txt'), 'r')
        label = labelFile.readline().strip()
        labelFile.close()
        
        # Get features
        afp.processFile(eng, os.path.join(dataPath, audioFile))
        feats = eng.readAllOutputs()
        keys = list(feats.keys())

        # Update the number of frames
        numFrames = len(feats.get(keys[0]))
        totalFrames += numFrames

        # Write features
        for frInd in range(numFrames):
            for feat in keys:
                for val in feats.get(feat)[frInd]:
                    featFile.write('%.8f ' % val)
            
            featFile.write('%d' % instrIndex.get(label))
            featFile.write('\n')

    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()


# MAIN

trainAudio = './IRMAS-Sample/Training/'
trainFeats = './trainFeatures.dat'
testAudio = './IRMAS-Sample/Testing/'
testFeats = './testFeatures.dat'
model = 'model.svm'

# Specify features
# Change the feature dimensions value in the functions above if these are changed
fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512 CepsNbCoeffs=13 FFTWindow=Hamming')
fp.addFeature('lpc: LPC blockSize=1024 stepSize=512 LPCNbCoeffs=5')

# Initialize yaafe tools
df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

# Write training features
instruments = writeTrainFeatures(trainAudio, trainFeats)

# Train the svm 
subprocess.call(['./trainSVM', '-multi', trainFeats, model])

# Write cross-validation features
writeTestData(testAudio, testFeats, instruments)

# Cross-validate the model
subprocess.call(['./testSVM', '-multi', model, testFeats])
