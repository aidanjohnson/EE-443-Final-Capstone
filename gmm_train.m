% Class 0: cello
% Class 1: saxophone
% Class 2: violin

clear all;
len = 512;
numFilter = 48;
numCoeff = 13;

feats = [];
audioFile = dir('*.wav'); 
mfcc = zeros(1, numCoeff);
sss = zeros(1, 4);
for k = 1:length(audioFiles)
  
  % Process file
  filename = myFiles(k).name;
  fprintf(1, 'Now reading %s\n', filename);
  [audio, fs] = audioread(filename);
  
  % Compute fft and mfcc for each frames
  for n = 1:floor(length(audio)/len)
     frame = audio((n - 1)*len + 1: n*len);
     mag = abs(fft(frame));
     
     % MFCC
     for m = 1:numCoeff
        mfcc(m) = GetCoefficient(mag, fs, 48, len, m);
     end
     
     % Shape statistics
     sss = GetShapeStatistics(mag);
     
     feats = [feats; [mfcc, sss]];
  end
end

save('features.dat', 'feats');

nc = 3;
GMModel = fitgmdist(trainData, nc);

save('GMModel.mat', 'GMModel');
