%% Attempt reachability

% model
betti_oo = importNetworkFromONNX("betti_oo.onnx");
net = matlab2nnv(betti_oo);

% Data
xb = readNPY('betti_input.npy');
xb = permute(xb, [3,4,2,1]);
yb = readNPY("betti_output.npy");
yb = permute(yb, [3,4,2,1]);

% inference
yo  = predict(betti_oo,xb);

% Are we importing the same model?
try
    EPS = 1e-3;
    assert(all(abs(yo - yb) <= EPS,'all')) % this fails
    disp("We are good with eps = "+string(EPS));
catch
    xxxx = abs(yo - yb);
    disp("Max Numerical error:  " + string(max(xxxx, [], 'all')));
    disp("Average error per pixel:   " + string(mean(xxxx, 'all')))
end

%% Reach
X
= ImageStar(xb, xb);

reachOptions.reachMethod = "relax-star-range";
reachOptions.relaxFactor = 1;

Y = net.reach(X, reachOptions);
[lb,ub] = Y.estimateRanges;


% Are we importing the same model?
EPS = 1e-2;
assert(all(lb <= yb+EPS,'all'), "Reachability not sound; output value found lower than lower bound")
assert(all(ub >= yb-EPS,'all'), "Reachability not sound; output value found larger than upper bound")
disp("We are good with eps = "+string(EPS));

%% Reach (proper bounds) % can't run it on local desktop
% Requested 92416x92416 (63.6GB) array exceeds maximum array size preference (62.7GB). This might cause MATLAB to become unresponsive.

epsilon = 0.0001;
X = ImageStar(xb-epsilon, xb+epsilon);

reachOptions.reachMethod = "relax-star-range";
reachOptions.relaxFactor = 1;

Y = net.reach(X, reachOptions);
[lb,ub] = Y.estimateRanges;


% Are we importing the same model?
EPS = 1e-2;
assert(all(lb <= yb+EPS,'all'), "Reachability not sound; output value found lower than lower bound")
assert(all(ub >= yb-EPS,'all'), "Reachability not sound; output value found larger than upper bound")
disp("We are good with eps = "+string(EPS));