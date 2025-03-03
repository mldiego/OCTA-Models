%% Can we get these pytorch models into NNV?


%% BETTI
% xb = readNPY('betti_input.npy');
% yb = readNPY("betti_output.npy");
% 
% modelfile = "betti_best.pt";
% betti = importNetworkFromPyTorch(modelfile);

%% Vanilla
% xv = readNPY('vanilla_input.npy');
% yv = readNPY("vanilla_output.npy");
% 
% modelfile = "vanilla_best.pt";
% vanilla = importNetworkFromPyTorch(modelfile);

%% Notes
% We cannot really use these...

% Convert them to onnx

% bettiO = importNetworkFromONNX("bettiO.onnx");
% vanillaO = importNetworkFromONNX("vanillaO.onnx");

% Same, we cannot use these...

% Exporting them using op_set=10
betti_oo = importNetworkFromONNX("betti_oo.onnx");
vanilla_oo = importNetworkFromONNX("vanilla_oo.onnx");

% YAY!! These work!!