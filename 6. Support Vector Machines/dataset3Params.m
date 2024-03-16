function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

minimumError = realmax;
fprintf('Initial prediction error calculated from the cross validation set: \n\n%f \n', minimumError);

# for c = [0.01,0.03,0.1,0.3,1,3,10,30,100,300]
#    for Sigma = [0.01,0.03,0.1,0.3,1,3,10,30,100,300]
values = [0.1,1];
for c = 1
   for Sigma = 0.1
        fprintf('Using a value of c =  %f \n',c);
        fprintf('Using a value of Sigma =  %f \n',Sigma);
        modelVal = svmTrain(Xval, yval, c, @(x11, x22) gaussianKernel(x11, x22, Sigma));
        # modelVal = svmTrain(Xval, yval, C, @linearKernel, 1e-3, 20);
        predictions = svmPredict(modelVal, Xval);
        predictionError = mean(double(predictions ~= yval));
        oldMinimumError = minimumError;
        if (predictionError < minimumError)
            minimumError = predictionError;
            C = c;
            sigma = Sigma;
            fprintf('Choosing a value of C =  %f \n', C);
            fprintf('Choosing a value of sigma =  %f \n', sigma);
        endif

        fprintf('Prediction error calculated from the cross validation set %f \n',predictionError);
        fprintf('Previous pPrediction error calculated from the cross validation set %f \n\n\n\n',oldMinimumError);

        #model = svmTrain(X, y, c, @(x11, x22) gaussianKernel(x11, x22, Sigma));
        #visualizeBoundary(X, y, model);
        #pause;
    endfor
endfor

fprintf('C calculated from the cross validation set %f \n',C);
fprintf('sigma calculated from the cross validation set %f \n',sigma);
fprintf('Minimum prediction error calculated from the cross validation set %f \n',minimumError);
% =========================================================================

end
