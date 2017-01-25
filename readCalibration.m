function [P, Tr_velo_to_cam, R_cam_to_rect] = readCalibration(calib_dir,img_idx,cam)

% load 3x4 projection matrix
P = dlmread(sprintf('%s/%06d.txt',calib_dir,img_idx),' ',0,1);

Tr_velo_to_cam = P(6,:);

R_cam_to_rect = P(5,1:9);

P = P(cam+1,:);
P = reshape(P ,[4,3])';
Tr_velo_to_cam = reshape(Tr_velo_to_cam ,[4,3])';
Tr_velo_to_cam = [Tr_velo_to_cam;0 0 0 1];
R_cam_to_rect = reshape(R_cam_to_rect ,[3,3])';

end