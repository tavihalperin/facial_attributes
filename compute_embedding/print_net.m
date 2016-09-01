% config.paths.net_path = 'data/vgg_face.mat';
% convNet = lib.face_feats.convNet(config.paths.net_path);
out = '';
for i = 1:length(convNet.net.layers)
    out = sprintf('%s\n%d %s (%s) ',out, i, convNet.net.layers{i}.type, convNet.net.layers{i}.name);
    if isfield(convNet.net.layers{i}, 'weights')
        out = sprintf('%s %s', out, num2str(size(convNet.net.layers{i}.weights{1})));
    end
    if isfield(convNet.net.layers{i}, 'pool')
        out = sprintf('%s %s', out, num2str(convNet.net.layers{i}.pool));
    end
end
disp(out);