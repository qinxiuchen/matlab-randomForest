function [ forest ] = randomForest( X, y, T, Test_X, Test_y )

N = length(y);
Test_N = length(Test_y);
%error_rate_total = 0;
%Error_stats = zeros(T, 1);
%for n=1:100
predict_y_test = zeros(Test_N, 1);
predict_y_train = zeros(N, 1);
trees = cell(T, 1);
for t = 1:T
    sampleIds = randsample(1:N, N, true);
    sample_X = X(sampleIds, :);
    sample_y = y(sampleIds);
    tree = decisionTreeTrain(sample_X, sample_y);
    trees{t} = tree;
    predict_y_test = predict_y_test + decisionTreePredict( Test_X, tree );
    predict_y_train = predict_y_train + decisionTreePredict( X, tree );
    %error_count = 0;
    %for i = 1:N
    %    if y(i) ~= predict_y(i)
    %        error_count = error_count + 1;
    %    end
    %end
    %error_rate = error_count / N;
    %Error_stats(t) = error_rate;
end
%forest.Error_stats = Error_stats;
forest.trees = trees;
error_count = 0;
for i=1:Test_N
    if predict_y_test(i) > 0
        predict_y_test(i) = 1;
    else
        predict_y_test(i) = -1;
    end
    if predict_y_test(i) ~= Test_y(i)
        error_count = error_count + 1;
    end
end
forest.error_rate_test = error_count / Test_N;

error_count = 0;
for i=1:N
    if predict_y_train(i) > 0
        predict_y_train(i) = 1;
    else
        predict_y_train(i) = -1;
    end
    if predict_y_train(i) ~= y(i)
        error_count = error_count + 1;
    end
end
forest.error_rate_train = error_count / N;

%error_rate_total = error_rate_total + error_count / length(Test_y);
%end
%forest.error_rate = error_rate_total/100;
end