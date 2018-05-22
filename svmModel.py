import os
import numpy as np
import yaafelib as yl
# import svmlight as sl

fp = yl.FeaturePlan(sample_rate=44100)
fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
# fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
# fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
fp.addFeature('sf: SpectralFlatness blockSize=1024')
fp.addFeature('sr: SpectralRolloff blockSize=1024')

df = fp.getDataFlow()
eng = yl.Engine()
eng.load(df)
afp = yl.AudioFileProcessor()

datapath = "/home/deniz/Documents/uw/7_2018sp/ece443/project/dataset/IRMAS-Sample/Training/"

for instrument in os.listdir(datapath):
    for audioFile in os.listdir(os.path.join(datapath, instrument)):
        afp.processFile(eng, os.path.join(datapath, instrument, audioFile))
        feats = eng.readAllOutputs()
