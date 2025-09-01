# 导入所需要的库
import cv2
import imageio
import numpy as np

def get_source_info_opencv(source_name):
    return_value = 0  
    try:
        cap = cv2.VideoCapture(source_name)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("width:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(width, height, fps, num_frames))
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("init_source:{} error. {}\n".format(source_name, str(e)))
        return_value = -1
    return return_value

for video_id in range(7, 8):  

    video_path = "/home/drink/SEED-DV/Video/" + str(video_id) + "th_10min.mp4"
      
    get_source_info_opencv(video_path)
    # 读取视频文件
    videoCapture = cv2.VideoCapture(video_path) 
    is_video = np.zeros(24*(8*60+40))
    print(is_video.shape)

    for i in range(40):
        is_video[i*(24*(13)):i*(24*(13))+3*24] = 0
        for j in range(5):
            is_video[i*(24*(13))+3*24+j*24*2:i*(24*(13))+3*24+j*24*2+24*2] = j+1
      
    #读帧
    k = 0
    i = -1
    while i < 12480:
        i += 1
        success, frame = videoCapture.read()
        frame = frame[..., ::-1] 
        if(is_video[i] == 0):
            continue
        all_frame = [cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)]
        while(i+1<12480 and is_video[i+1] == is_video[i]):
            i += 1
            success, frame = videoCapture.read()
            frame = frame[..., ::-1] 
            all_frame.append(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR))
        gif_frame = []
        for j in range(0, 48, 8):
            gif_frame.append(all_frame[j])    
        print("k = ", k, len(gif_frame))
        imageio.mimsave('/home/drink/SEED-DV/Video_Gif/Block' + str(video_id) + '/'+str(k)+'.gif', gif_frame, 'GIF', duration=0.33333)  
        
        k += 1