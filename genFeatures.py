import os
import yaafelib as yl

fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
# fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
# fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
# fp.addFeature('sf: SpectralFlatness blockSize=1024')
# fp.addFeature('sr: SpectralRolloff blockSize=1024')

df = fp.getDataFlow()
engine = yl.Engine()
engine.load(df)
afp = yl.AudioFileProcessor()
outpath = "features/"
afp.setOutputFormat('csv', outpath, {'Precision':'8'})

datapath = "./IRMAS-Sample/Training/"
instruments = os.listdir(datapath)

for instr in instruments:
    for audioFile in os.listdir(os.path.join(datapath, instr)):
        afp.processFile(engine, os.path.join(datapath, instr, audioFile))



