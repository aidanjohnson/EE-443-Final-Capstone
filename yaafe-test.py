#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 08:47:24 2018

@author: aidan
"""

# import numpy as np
from yaafelib import *
import pyglet
import sklearn

audiofile = 'IRMAS-TrainingData/pia/254__[pia][nod][cla]1406__2.wav'

fp = FeaturePlan(sample_rate=44100)
fp.loadFeaturePlan('featureplan.txt')
# All the features below are listed line-by-line in featureplan.txt:
# fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
# fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
# fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
# fp.addFeature('sf: SpectralFlatness blockSize=1024')
# fp.addFeature('sr: SpectralRolloff blockSize=1024')
# fp.addFeature('ss: SpectralSlope')
# fp.addFeature('obsi: OBSI')
# fp.addFeature('tss: TemporalShapeStatistics')
# fp.addFeature('zcr: ZCR')
# fp.addFeature('sv: SpectralVariation')
# fp.addFeature('sss: SpectralShapeStatistics')

df = fp.getDataFlow()
df.display()

engine = Engine()
engine.load(df)

afp = AudioFileProcessor()
afp.processFile(engine,audiofile)

feats = engine.readAllOutputs()

afp.setOutputFormat('csv','output',{'Precision':'8'})
afp.processFile(engine,audiofile)

# audio = np.random.randn(1,100000)
# feats = engine.processAudio(audio)

song = pyglet.media.load(str(audiofile))
song.play()
