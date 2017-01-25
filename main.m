function main()

% specify training data directory path
root_dir = './data_object_velodyne/training/velodyne/';

% specify training label directory path
label_dir = './training/label_2/';

% specify output directory path
out_dir = './grid_files/';


% specify resolution of grid
res = 10;



for frame = 1:1500
    % load velodyne points
    fid = fopen(sprintf('%s%06d.bin',root_dir,frame),'rb');
    velo = fread(fid,[4 inf],'single')';
    % remove every 5th point for display speed
    %velo = velo(1:5:end,:);
    fclose(fid);
    
    % remove all points behind image plane (approximation
    idx = velo(:,1)<5;
    velo(idx,:) = [];
    
    % convert pcl to grid file with the specified resolution
    grid = pcltogrid(velo, res);
    
    filename = sprintf('%s%06d.csv',out_dir,frame);
    csvwrite(filename, grid);
    
    train_label = sprintf('%s%06d.txt',label_dir,frame);
    copyfile(train_label, out_dir);
end


end