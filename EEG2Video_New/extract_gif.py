import cv2
import imageio
import numpy as np

def ordinal(n: int) -> str:
    """return 1st, 2nd, 3rd, 4th …"""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

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

for video_id in range(1, 8):  
    video_path = "SEED-DV/Video/" + ordinal(video_id) + "_10min.mp4"
      
    get_source_info_opencv(video_path)
    
    videoCapture = cv2.VideoCapture(video_path) 
    is_video = np.zeros(24*(8*60+40)) # 12480 frames, 24 fps, 8 minutes + 40 seconds

    # mark which frames belong to which clip (5 clips per concept, 40 concepts)
    for i in range(40):
        is_video[i*(24*(13)):i*(24*(13))+3*24] = 0  # 3-second rest
        for j in range(5):
            is_video[i*(24*(13))+3*24+j*24*2:i*(24*(13))+3*24+j*24*2+24*2] = j+1  # 2-second clip
            
    # read frames
    k = 0  # gif index
    i = -1
    while i < 12480:
        i += 1
        success, frame = videoCapture.read()  # grab one frame
        frame = frame[..., ::-1]  # change the channel order from BGR to RGB
        if(is_video[i] == 0):  # skip non-clip frames
            continue
        all_frame = [cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)]  # store first frame
        while(i+1<12480 and is_video[i+1] == is_video[i]):  # collect same-label frames
            i += 1
            success, frame = videoCapture.read()
            frame = frame[..., ::-1] 
            all_frame.append(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR))
        gif_frame = []
        for j in range(0, 48, 8):  # pick every 8th frame → 6 frames total
            gif_frame.append(all_frame[j])    
        print("k = ", k)
        imageio.mimsave('SEED-DV/Video_Gif/Block' + str(video_id) + '/'+str(k)+'.gif', gif_frame, 'GIF', duration=0.33333)  # save gif
        k += 1
