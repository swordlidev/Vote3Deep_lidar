function grid = pcltogrid(velo, res)
clc;

maxX = max(velo(:,1));
minX = min(velo(:,1));


maxY = max(velo(:,2));
minY = min(velo(:,2));

maxZ = max(velo(:,3));
minZ = min(velo(:,3));

xWidth = (maxX-minX)/res;
yWidth = (maxY-minY)/res;
zWidth = (maxZ-minZ)/res;

[m,n] = size(velo);

mat = zeros(m,n+3);
for i=1:m
    
    xBin = ceil((velo(i,1) - minX)/xWidth);
    yBin = ceil((velo(i,2) - minY)/yWidth);
    zBin = ceil((velo(i,3) - minZ)/zWidth);
    if(xBin == 0)
        xBin = 1;
    end
    
    if(yBin == 0)
        yBin = 1;
    end
    
    if(zBin == 0)
        zBin = 1;
    end
    
    arr = [xBin yBin zBin velo(i,:)];
    mat(i,:) = arr;
end
mat = sortrows(mat, [1,2,3]);

grid = [];
prevX = mat(1,1);
prevY = mat(1,2);
prevZ = mat(1,3);

data = [];
for i=1:m
    if(prevZ ~= mat(i,3))
        %try
        if(isempty(data))
            % Ensure if the grid has only one data point 
            grid = [grid ; cat(2, mat(i-1,:), [0 1])];
        else
            grid = [grid ; prevX prevY prevZ mean(data(:,1)) mean(data(:,2)) mean(data(:,3)) mean(data(:,4)) var(data(:,4)) 1];
            data = [];
        end
        
        %catch
        %display('Error!');
    else
        data = [data; mat(i,4) mat(i, 5) mat(i, 6) mat(i,7)];
    end
    prevX = mat(i,1);
    prevY = mat(i,2);
    prevZ = mat(i,3);
    
end

end