% Class 0: cello
% Class 1: saxophone
% Class 2: violin

clear all;

data = importdata('training.dat');
[samps, dims] = size(data);
nc = 3; % Number of classes

classes = data(:, dims);
% idx = [sum(classes == 0), sum(classes == 1), sum(classes == 2)];
idx = sum(repmat(classes, 1, nc) == 0:nc - 1);
cel = data(1 : idx(1), 1:dims - 1);
sax = data(idx(1) + 1 : sum(idx(1:2)), 1:dims - 1);
vio = data(sum(idx(1:2)) + 1 : sum(idx), 1:dims - 1);

trainData = data(:, dims - 1);
GMModel = fitgmdist(dims, nc);

save('GMModel.mat', 'GMModel');


