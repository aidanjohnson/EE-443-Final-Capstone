import os
import subprocess
import yaafelib as yl


def getInstruments(dataPath):

    # Instruments to ignore
    ignInstr = ['voi', 'gel', 'gac', 'org', 'tru', 'flu'] 

    # This leaves cel, cla, pia, sax, vio
    ignInstr.append('pia')
    ignInstr.append('vio') 
    ignInstr.append('sax')

    # Assign class indices of the instruments
    instrList = [n for n in os.listdir(dataPath) if not n.endswith('.txt') and n not in ignInstr]
    instrIndex = dict(zip(instrList, range(len(instrList))))
    
    return instrIndex


def writeTrainFeatures(dataPath, featPath, instrIndex):
    
    featFile = open(featPath, 'w')
    featFile.write('             ') # Place holder for top line that is written last
    totalFrames = 0 # The total number of frames processed
    limit = 60     # The number of audio files for each instrument that's used

    for instr in (fn for fn in instrIndex.keys() if not fn.endswith('.txt')):

        count = 0
        subStr = '[' + instr + ']' + '[nod][cla]'

        for audioFile in (fn for fn in os.listdir(os.path.join(dataPath, instr)) if subStr in fn):
            if (count < limit):

                count += 1

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
        otherInstr = labelFile.readline().strip()
        labelFile.close()

        if (label in instrIndex and not otherInstr):
            
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
trainAudio = '/home/deniz/Documents/IRMAS/TrainingData/'
trainFeats = './MultiModel/trainFeatures.dat'
testAudio = '/home/deniz/Documents/IRMAS/TestingData/'
testFeats = './MultiModel/testFeatures.dat'
model = './MultiModel/model.svm'

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

# Remove previous model files

# Write training features
print('\nExtracting training features\n')
writeTrainFeatures(trainAudio, trainFeats, instruments)

# Train the svm 
print('\nTraining SVM\n')
subprocess.call(['./trainSVM', '-multi', trainFeats, model])

# Write the cross-validation features
print('\nExtracting testing features\n')
writeTestFeatures(testAudio, testFeats, instruments)

# Test the svm on the cross-validation features
print('\nTesting SVM on cross-validation features\n')
subprocess.call(['./testSVM', '-multi', model, testFeats])

# Test the svm on the training features
print('\nTesting SVM on training features\n')
subprocess.call(['./testSVM', '-multi', model, trainFeats])
