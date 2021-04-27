framesPath = 'output/output0402_/Ucar_test_0317/';%图像序列所在路径，同时要保证图像大小相同
videoName = 'output/output0402_/Ucar_test_0317/output.mp4';%表示将要创建的视频文件的名字
fps = 30; %帧率
startFrame = 1; %从哪一帧开始
endFrame = 429; %哪一帧结束

if(exist('videoName','file'))
delete output.mp4
end

%生成视频的参数设定
aviobj=VideoWriter(videoName); %创建一个avi视频文件对象，开始时其为空
aviobj.FrameRate=fps;

open(aviobj);%Open file for writing video data
%读入图片
list = dir(strcat(framesPath,'*.png'));
for i=startFrame:endFrame
%fileName=sprintf('%04d',i); %根据文件名而定 我这里文件名是0001.jpg 0002.jpg ….
fileName=list(i).name;
display(fileName)
frames=imread([framesPath,fileName]);
frames=imresize(frames,[369,492]);
writeVideo(aviobj,frames);
end
close(aviobj);% 关闭创建视频