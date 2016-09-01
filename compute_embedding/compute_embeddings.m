net_path = 'data/vgg_face.mat';
face_model_path = 'data/face_model.mat';
data_path = '../../';

faceDet = lib.face_detector.dpmCascadeDetector(face_model_path);
convNet = lib.face_feats.convNet(net_path);

fileID = fopen([data_path 'EVAL/list_eval_partition.txt']);
C = textscan(fileID,'%s');
fclose(fileID);
C = C{1};
C = C(1:2:end);
img_folder = [data_path 'img_celeba/img_align_celeba/'];
out_folder = '../embeddings/';

non_face_counter = 0;
layers_to_keep = [32, 35]; %32 is FC6, 35 is FC7
for l=1:numel(convNet.net.layers)
    convNet.net.layers{l}.rememberOutput = any(ismember(layers_to_keep,l));
end
all_embeddings = containers.Map();
for pos = 1:length(C)
    img_file = C{pos};
    im = imread([img_folder img_file]);
    det = faceDet.detect(img);
    if numel(det) == 0
        non_face_counter =  non_face_counter + 1;
    else
        my_crop = myFaceCrop.crop(im,det(1:4,1));
        all_embeddings(img_file(1:end-4)) = convNet.simpleNN(my_crop);        
    end

    if mod(pos,100)==1
        fprintf('%d ',pos);
    end
    if mod(pos,1000)==1
        fprintf('\n');
    end

end

all_keys = keys(all_embeddings);
save('../good_keys.mat', 'all_keys');
embeddings_32 = zeros(numel(all_keys), 4096);
embeddings_35 = zeros(numel(all_keys), 4096);
for i = 1:numel(all_keys)
    key = all_keys(i);
    v =  all_embeddings(key{1});
    embeddings_32 (i, :) = v{1};
    embeddings_35 (i, :) = v{2};
end
out_file_32 = [out_folder 'emb_32.mat'];
out_file_35 = [out_folder 'emb_35.mat'];
% save FC6, in small parts to avoid too big variables
part1 = embeddings_32(1:50000,:);  %#ok<NASGU>
part2 = embeddings_32(50001:100000,:);  %#ok<NASGU>
part3 = embeddings_32(100001:150000,:);  %#ok<NASGU>
part4 = embeddings_32(150001:end,:);  %#ok<NASGU>
save('../embeddings_32_split.mat','part1', 'part2', 'part3', 'part4');
clear('part1', 'part2', 'part3', 'part4');
% save FC7
part1 = embeddings_35(1:50000,:);
part2 = embeddings_35(50001:100000,:);
part3 = embeddings_35(100001:150000,:);
part4 = embeddings_35(150001:end,:);
save('../embeddings_35_split.mat','part1', 'part2', 'part3', 'part4');

non_face_counter %#ok<NOPTS>