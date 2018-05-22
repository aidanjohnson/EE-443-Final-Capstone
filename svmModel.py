import os
import yaafelib as yl

fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512')
# fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
# fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
# fp.addFeature('sf: SpectralFlatness blockSize=1024')
# fp.addFeature('sr: SpectralRolloff blockSize=1024')

df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

datapath = './IRMAS-Sample/Training/'
featFile = open('features.dat', 'w')
instrInd = 0 # This is the index of the instruments used by the classifier

for instr in os.listdir(datapath):
    for audioFile in os.listdir(os.path.join(datapath, instr)):
        
        # Get features
        afp.processFile(eng, os.path.join(datapath, instr, audioFile))
        feats = eng.readAllOutputs()

        # Write features
        for feat in feats.keys():
            for frame in feats.get(feat):
                for val in frame:
                    featFile.write('%.8f ' % val)
                featFile.write('%d' % instrInd)
                featFile.write('\n')
    instrInd += 1