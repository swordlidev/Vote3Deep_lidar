cam       = 2;
frame     = 20;

% compute projection matrix velodyne->image plane
R_cam_to_rect = eye(4);
[P, Tr_velo_to_cam, R] = readCalibration('./data_object_calib/training/calib',frame,cam)
R_cam_to_rect(1:3,1:3) = R;
P_velo_to_img = P*R_cam_to_rect*Tr_velo_to_cam;


% load and display image
% img = imread(sprintf('D:/Shared/training/image_2/%06d.png',frame));
% fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
% imshow(img); hold on;

% load velodyne points
fid = fopen(sprintf('./data_object_velodyne/training/velodyne/%06d.bin',frame),'rb');
velo = fread(fid,[4 inf],'single')';
% remove every 5th point for display speed
velo = velo(1:5:end,:);
fclose(fid);

% remove all points behind image plane (approximation
idx = velo(:,1)<5;
velo(idx,:) = [];

project to image plane (exclude luminance)
velo_img = project(velo(:,1:3),P_velo_to_img);

% plot points
cols = jet;
for i=1:size(velo_img,1)
    col_idx = round(64*5/velo(i,1));
    plot(velo_img(i,1),velo_img(i,2),'o','LineWidth',4,'MarkerSize',1,'Color',cols(col_idx,:));
end