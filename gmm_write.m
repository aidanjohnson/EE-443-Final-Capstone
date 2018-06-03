clear all;

load('GMModel.mat');

means = GMModel.mu;
weights = GMModel.PComponents;
vars = GMModel.Sigma;

GMMmeans = zeros(1,size(means,1)*size(means,2)); % 1D means
GMMvariances = zeros(1,size(vars,3)*size(vars,2)); % 1D means

for k = 1:size(means,1)
    GMMmeans(size(means,2)*(k-1)+1:size(means,2)*k) = means(k,:);
end

GMMweights = weights;

for k = 1:size(vars,3)
    GMMvariances(size(vars,2)*(k-1)+1:size(vars,2)*k) = diag(vars(:,:,k));
end

printTXT('GMMmeans.txt',GMMmeans);
printTXT('GMMweights.txt',GMMweights);
printTXT('GMMvariances.txt',GMMvariances);

function [] = printTXT(name, out)
    fileID = fopen(name,'w');
    fprintf(fileID,'{%.8f',out(1));
    if length(out) > 1
        fprintf(fileID,',%.8f',out(2:end));
    end
    fprintf(fileID,'}');
end