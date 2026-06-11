clc;
clear;
close all;

%% Load Dataset
data = readtable('stroke_dataset.csv');

%% Data Preprocessing

% Fill missing BMI values
data.bmi = fillmissing(data.bmi,'mean');

% Convert categorical variables
data.gender = grp2idx(categorical(data.gender));
data.ever_married = grp2idx(categorical(data.ever_married));
data.work_type = grp2idx(categorical(data.work_type));
data.Residence_type = grp2idx(categorical(data.Residence_type));
data.smoking_status = grp2idx(categorical(data.smoking_status));

%% Input Features and Target
X = data{:,1:end-1};
Y = data.stroke;

%% Normalize Features
X = normalize(X);

%% Train-Test Split
cv = cvpartition(Y,'HoldOut',0.2);

XTrain = X(training(cv),:);
YTrain = Y(training(cv));

XTest = X(test(cv),:);
YTest = Y(test(cv));

%% ==========================
% Medium Tree Classifier
%% ==========================

treeModel = fitctree(XTrain,YTrain,...
    'MaxNumSplits',20);

treePred = predict(treeModel,XTest);

treeAccuracy = sum(treePred == YTest)/length(YTest)*100;

fprintf('\nMedium Tree Accuracy = %.2f%%\n',treeAccuracy);

%% Confusion Matrix
figure;
confusionchart(YTest,treePred);
title('Medium Tree Confusion Matrix');

%% ==========================
% Coarse Gaussian SVM
%% ==========================

svmModel = fitcsvm(XTrain,YTrain,...
    'KernelFunction','gaussian',...
    'KernelScale','auto',...
    'BoxConstraint',1);

svmPred = predict(svmModel,XTest);

svmAccuracy = sum(svmPred == YTest)/length(YTest)*100;

fprintf('Coarse Gaussian SVM Accuracy = %.2f%%\n',svmAccuracy);

%% Confusion Matrix
figure;
confusionchart(YTest,svmPred);
title('Coarse Gaussian SVM Confusion Matrix');

%% ==========================
% Accuracy Comparison Graph
%% ==========================

accuracyValues = [treeAccuracy svmAccuracy];

figure;
bar(accuracyValues);

set(gca,'XTickLabel',{'Medium Tree','Coarse Gaussian SVM'});
ylabel('Accuracy (%)');
title('Classifier Accuracy Comparison');

%% ==========================
% Performance Metrics
%% ==========================

cm = confusionmat(YTest,treePred);

TP = cm(2,2);
TN = cm(1,1);
FP = cm(1,2);
FN = cm(2,1);

Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
F1 = 2*((Precision*Recall)/(Precision+Recall));

fprintf('\nPrecision = %.4f\n',Precision);
fprintf('Recall    = %.4f\n',Recall);
fprintf('F1 Score  = %.4f\n',F1);
