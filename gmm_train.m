% Class 0: cello
% Class 1: saxophone
% Class 2: violin

clear all;

len = 512;
numFilter = 48;
numCoeff = 13;

instr = {'cel', 'sax', 'vio'};
dataPath = '/home/deniz/Documents/IRMAS/TrainingData/'; 
featPath = '/home/deniz/Documents/uw/7_2018sp/ece443/project/features.dat';

mfcc = zeros(1, numCoeff);
sss = zeros(1, 4);
fid = fopen(featPath, 'w');
for l = 1:length(instr)

    fprintf('Extractin features for %s\n', instr{l});
    audioFiles = dir(fullfile(dataPath, instr{l}, '*.wav'));

    for k = 1:length(audioFiles)
        if (contains(audioFiles(k).name, '[nod]'))
            % Process files
            filename = audioFiles(k).name;
            fprintf('Processing %s\n', filename);
            [audio, fs] = audioread(fullfile(audioFiles(k).folder, audioFiles(k).name));
            
            % Compute features for each frame
            for n = 1:floor(length(audio)/len)
                
                frame = audio((n - 1)*len + 1 : n*len);
                mag = abs(fft(frame));
                
                % MFCC
                for m = 1:numCoeff
                    mfcc(m) = GetCoefficient(mag, fs, 48, len, m);
                end
                
                % Shape statistics
                sss = GetShapeStatistics(mag);
                
                feats = [mfcc, sss];
                fprintf(fid, '%.8f ', feats);
                fprintf(fid, '\n');
            end
        else
            fprintf('Ignored %s\n', audioFiles(k).name);
        end
    end
end

fprintf('Completed feature extraction\n');

% nc = 3;

% GMModel = fitgmdist(feats, nc);

% save('GMModel.mat', 'GMModel');