% Copyright for original version 
%  Copyright (c) 2015, Omkar Parkhi, Elliot Crowley 
%  All rights reserved.

classdef myFaceCrop < handle
	
	methods(Static)

	function crop = crop(img,box)
        extend = 0.1;
		%make square

        img_size = size(img);
        img_size = img_size(1:2);
        ext_size = ceil(img_size*(1+extend))+30;
        ext_img =  zeros(ext_size(1),ext_size(2),3,'uint8');
        s  = round((ext_size-img_size)/2);
        ext_img(s(1):s(1)+img_size(1)-1, s(2):s(2)+img_size(2)-1, :) = img;

        ext_img(1:s(1),:,:) = repmat(ext_img(s(1),:,:), s(1), 1); %TODO improve efficiency
        ext_img(s(1)+img_size(1)-1:end,:,:) = repmat(ext_img(s(1)+img_size(1)-1,:,:), ext_size(1)-s(1)-img_size(1)+2, 1);
        ext_img(:,1:s(2),:) = repmat(ext_img(:,s(2),:), 1, s(2)); %TODO improve efficiency
        ext_img(:,s(2)+img_size(2)-1:end,:) = repmat(ext_img(:,s(2)+img_size(2)-1,:), 1, ext_size(2)-s(2)-img_size(2)+2);

        img = ext_img;
        
        box([1,3]) = box([1,3]) + s(2) - 1;
        box([2,4]) = box([2,4]) + s(1) - 1;       

        width = round(box(3)-box(1));
		height = round(box(4)-box(2));

		length = (width + height)/2;

		centrepoint = [round(box(1)) + width/2 round(box(2)) + height/2];
		x1= centrepoint(1) - round((1+extend)*length/2);
		y1= centrepoint(2) - round((1+extend)*length/2);
		x2= centrepoint(1) + round((1+extend)*length/2);
		y2= centrepoint(2) + round((1+extend)*length/2);

        img = img(round(y1):round(y2),round(x1):round(x2),:);
		sizeimg = size(img);

        newdim = 224;
		crop = imresize(img,(newdim/sizeimg(1)));
		
	end

	end
end
