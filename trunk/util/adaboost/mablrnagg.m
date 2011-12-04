function [learner, weight] = mablrnagg(learners, weights)

    learner.predict = 'mabmultipredict';
    agg_model.learners = learners;
    agg_model.weights = weights;
    learner.model = agg_model;
    weight = 1;
    
end
