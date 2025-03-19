%% Attempt reachability

% model
% betti_oo = importNetworkFromONNX("vanilla_oo.onnx");
load("vanilla_best.mat");
net = matlab2nnv(betti_oo);

% Data
% x = readNPY('vanilla_input.npy');
% x = permute(x, [3,4,2,1]);
% y = readNPY("vanilla_output.npy");
% y = permute(y, [3,4,2,1]);
load("vanilla_traces.mat");

% inference
% yo  = predict(betti_oo,xb);
% 
% % Are we importing the same model?
% try
%     EPS = 1e-3;
%     assert(all(abs(yo - yb) <= EPS,'all')) % this fails
%     disp("We are good with eps = "+string(EPS));
% catch
%     xxxx = abs(yo - yb);
%     disp("Max Numerical error:  " + string(max(xxxx, [], 'all')));
%     disp("Average error per pixel:   " + string(mean(xxxx, 'all')))
% end

%% Reach
% X = ImageStar(xb, xb);
% 
% reachOptions.reachMethod = "relax-star-range";
% reachOptions.relaxFactor = 1;
% 
% Y = net.reach(X, reachOptions);
% [lb,ub] = Y.estimateRanges;


% Are we importing the same model?
% EPS = 1e-2;
% assert(all(lb <= yb+EPS,'all'), "Reachability not sound; output value found lower than lower bound")
% assert(all(ub >= yb-EPS,'all'), "Reachability not sound; output value found larger than upper bound")
% disp("We are good with eps = "+string(EPS));

%% Reach (proper bounds) % can't run it on local desktop
% Requested 92416x92416 (63.6GB) array exceeds maximum array size preference (62.7GB). This might cause MATLAB to become unresponsive.
% 
% epsilon = 0.0001;
% X = ImageStar(xb-epsilon, xb+epsilon);
% 
% reachOptions.reachMethod = "relax-star-range";
% reachOptions.relaxFactor = 1;
% 
% Y = net.reach(X, reachOptions);
% [lb,ub] = Y.estimateRanges;


% Are we importing the same model?
% EPS = 1e-2;
% assert(all(lb <= yb+EPS,'all'), "Reachability not sound; output value found lower than lower bound")
% assert(all(ub >= yb-EPS,'all'), "Reachability not sound; output value found larger than upper bound")
% disp("We are good with eps = "+string(EPS));


%% Let's evaluate scalability with epsilon size and method

rF = [0,0.25,0.5,0.75,1];
EPSILON = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01];

rT = zeros(length(rF), length(EPSILON));

reachOptions = struct;
reachOptions.reachMethod = "relax-star-range";

for i=1:length(rF)
    for j=1:length(epsilon)
        reachOptions.relaxFactor = rF(i);
        epsilon = EPSILON(j);
        X = ImageStar(x-epsilon, x+epsilon);

        t = tic;
        net.reach(X, reachOptions);
        rT(i,j) = toc(t);

    end
end

save("vanilla_rT.mat", "rT")
        