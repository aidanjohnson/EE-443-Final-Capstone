import os
import subprocess
import yaafelib as yl

fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512 CepsNbCoeffs=13 FFTWindow=Hamming')
fp.addFeature('lpc: LPC blockSize=1024 stepSize=512 LPCNbCoeffs=5')

df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

datapath = './IRMAS-Sample/Training/'
featFile = open('features.dat', 'w')
featFile.write('1234578')    # Place holder for top line that is written last
instrInd = 0    # The index of the instruments used by the classifier
totalFrames = 0 # The total number of frames processed
dimensions = 18 # The sum of the dimensiosn of the features

for instr in os.listdir(datapath):
    for audioFile in os.listdir(os.path.join(datapath, instr)):
        
        # Get features
        afp.processFile(eng, os.path.join(datapath, instr, audioFile))
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
            featFile.write('%d' % instrInd)
            featFile.write('\n')
    
    instrInd += 1

# Write the number of frames and dimensions
featFile.seek(0)
featFile.write('%d %d\n' % (totalFrames, dimensions + 1))
featFile.close()

# Train svm
subprocess.call(['./trainSVM', '-multi', 'features.dat', 'model.svm'])

# Validate the model against test data

