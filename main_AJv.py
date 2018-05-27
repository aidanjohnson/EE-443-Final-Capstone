import os
import subprocess
import yaafelib as yl


def getInstruments(dataPath):

    # Include instruments from: cel, cla, flu, gac, gel, org, pia, sax, tru, vio
    instrList = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio']

    # Return instruments and class numbers
    return dict(zip(instrList, range(len(instrList))))
    
def writeFeatures(dataPath, featPath, instrIndex):
    
    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    numFiles = dict(zip(instrIndex.keys(), [0]*len(instrIndex)))

    for instr in (fn for fn in instrIndex.keys() if not fn.endswith('.txt')):

        subStr = '[' + instr + ']'

        for audioFile in (fn for fn in os.listdir(os.path.join(dataPath, instr)) if subStr in fn):



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
                
                # Write the instrument label
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

trainAudio = './IRMAS-Data/Training'
trainFeats = './trainFeatures.dat'
testAudio = './IRMAS-Data/Testing'
testFeats = './testFeatures.dat'
model = './model.svm'

# Get the instruments and their class indices
instruments = getInstruments(trainAudio)

# Specify features
fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC CepsNbCoeffs=13 FFTWindow=Hamming')
fp.addFeature('sss: SpectralShapeStatistics FFTWindow=Hamming')
# fp.addFeature('obsi: OBSI blockSize=1024 stepSize=512 FFTWindow=Hamming')
# Dimensions: mfcc +13, sss +4, obsi +10
dimensions = 17 # The sum of the dimensions of the features

# Initialize yaafe tools
df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

# Remove previous model files
for k in range(len(instruments)):
    classFile = model + '.' + str(k)
    if (os.path.isfile(classFile)):
        os.remove(classFile)

# Write training features
print('\nExtracting training features\n')
numTrainFiles = writeFeatures(trainAudio, trainFeats, instruments)

# Display the number of training files used
print('Number of training files used: ' + str(numTrainFiles) + '\n')

# Train the svm 
print('\nTraining SVM\n')
subprocess.call(['./SVMTorch/SVMTorch', '-multi', trainFeats, model])

# Write the cross-validation features
print('\nExtracting testing features\n')
numTestFiles = writeFeatures(testAudio, testFeats, instruments)

# Display the number of testing files used
print('Number of testing files used: ' + str(numTestFiles) + '\n')

# Test the svm on the cross-validation features
print('\nTesting SVM on cross-validation features\n')
subprocess.call(['./SVMTorch/SVMTest', '-multi', model, testFeats])

# Test the svm on the training features
print('\nTesting SVM on training features\n')
subprocess.call(['./SVMTorch/SVMTest', '-multi', model, trainFeats])
