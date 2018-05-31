import os
import subprocess
import yaafelib as yl


def getInstruments(dataPath):

    # Include instruments from: cel, cla, flu, gac, gel, org, pia, sax, tru, vio
    # instrList = ['cel', 'cla', 'sax', 'vio']
    instrList = ['cel', 'sax', 'vio']

    # Return instruments and class numbers
    return dict(zip(instrList, range(len(instrList))))


def writeTrainFeatures(dataPath, featPath, instrIndex):
    
    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    limit = 300     # The number of audio files for each instrument that's used
    numFiles = dict(zip(instrIndex.keys(), [0]*len(instrIndex)))

    for instr in (fn for fn in instrIndex.keys() if not fn.endswith('.txt')):

        subStr = '[' + instr + '][nod]'

        for audioFile in (fn for fn in os.listdir(os.path.join(dataPath, instr)) if subStr in fn):
            if (numFiles[instr] < limit):
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

                # Update the number of files processed for the instrument
                numFiles[instr] += 1
    
    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()

    return numFiles


def writeTestFeatures(dataPath, featPath, instrIndex):

    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    limit = 60      # The number of files used for each instrument
    numFiles = dict(zip(instrIndex.keys(), [0]*len(instrIndex)))
    
    for audioFile in (fn for fn in os.listdir(dataPath) if fn.endswith('.wav')):
        
        # Get the label
        subStr = audioFile.split('.')
        filename = ".".join(subStr[0 : len(subStr) - 1])
        instrFile = open(os.path.join(dataPath, filename + '.txt'), 'r')
        instr = instrFile.readline().strip()
        otherInstr = instrFile.readline().strip()
        instrFile.close()

        if (instr in instrIndex and not otherInstr):
            if (numFiles[instr] < limit):

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
                    
                    featFile.write('%d' % instrIndex.get(instr))
                    featFile.write('\n')
            
                # Update the number of files processed for the instrument
                numFiles[instr] += 1

    # Write the number of frames and dimensions
    featFile.seek(0)
    featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
    featFile.close()

    return numFiles


# Main

# trainAudio = '/home/deniz/Documents/IRMAS/IRMAS-Sample/Training/'
trainAudio = '/home/deniz/Documents/IRMAS/TrainingData/'
trainFeats = './MultiModel/CelClaSaxtrainFeatures.dat'
testAudio = '/home/deniz/Documents/IRMAS/TestingData/'
testFeats = './MultiModel/CelSaxVio/testFeatures.dat'
model = './MultiModel/CelSaxVio/model.svm'

# Get the instruments
instruments = getInstruments(trainAudio)

# Specify features
fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512 CepsNbCoeffs=13 FFTWindow=Hamming')
fp.addFeature('sss: SpectralShapeStatistics blockSize=1024 stepSize=512 FFTWindow=Hamming')
# fp.addFeature('obsi: OBSI blockSize=1024 stepSize=512 FFTWindow=Hamming')
# Dimensions: mfcc +13, sss +4, obsi +10
dimensions = 17 # The sum of the dimensions of the features

# Initialize yaafe tools
df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

# # Remove previous model files
# for k in range(len(instruments)):
#     classFile = model + '.' + str(k)
#     if (os.path.isfile(classFile)):
#         os.remove(classFile)

# # Write training features
# print('\nExtracting training features\n')
# numTrainFiles = writeTrainFeatures(trainAudio, trainFeats, instruments)

# # Train the svm 
# print('\nTraining SVM\n')
# subprocess.call(['./trainSVM', '-multi', trainFeats, model])

# Write the cross-validation features
print('\nExtracting testing features\n')
numTestFiles = writeTestFeatures(testAudio, testFeats, instruments)

# Test the svm on the cross-validation features
print('\nTesting SVM on cross-validation features\n')
subprocess.call(['./testSVM', '-multi', model, testFeats])

# # Test the svm on the training features
# print('\nTesting SVM on training features\n')
# subprocess.call(['./testSVM', '-multi', model, trainFeats])

# Display the number of training and testing files used
# print('Number of training files used: ' + str(numTrainFiles) + '\n')
print('Number of testing files used: ' + str(numTestFiles) + '\n')

