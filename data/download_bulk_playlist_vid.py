import os

ls_animated = '''
https://www.youtube.com/playlist?list=PLb62x91woEVNEO-nvmkgb6FpoAiH8eJ6Q
https://www.youtube.com/playlist?list=PLMBvr_k9eOnLdZC7XtMGpRFjj1ZO-X8O1
https://www.youtube.com/playlist?list=PLMBvr_k9eOnKRSpRTxoxj0QdRZb2lLfbD
https://www.youtube.com/playlist?list=PLJ4ZGUnQVwEJifVRJMxgiMcecflFf4mpY
https://www.youtube.com/playlist?list=PLnacEGPyd1mdvWuJRPs89QZp8yZxUcr6Q
https://www.youtube.com/playlist?list=PLJmPvVPsdAfuRjp33u3ip-Vm_2OieSNsi
https://www.youtube.com/playlist?list=PLRe9ARNnYSY41I4NXMtfHQ2HN2wap_YtX
https://www.youtube.com/playlist?list=PLvpJuQwm0QUxu2S1_qa273S7xGVPHF0O4
https://www.youtube.com/playlist?list=PLcdMYnP3Xox3G-uOgPZKXgvpXBjGeKAOa
https://www.youtube.com/playlist?list=PLQQQDDJmI-R4VQX8c65-8YEPUhgwS_Gd-
https://www.youtube.com/playlist?list=PLPCHPbv2gN2kAL0RnVl1vqr-8-48rHeF9
https://www.youtube.com/playlist?list=PLznZioURnjV-Wdscv9JKn4bLAKl87hYqJ
https://www.youtube.com/playlist?list=PLZCt7u9QT1Nx5JS782_mO5gGsaszly2Gh
https://www.youtube.com/playlist?list=PLAt8e3KL46sNK4Dl9ybiez1-6sBaIDJx4
https://www.youtube.com/playlist?list=PL5vdhFFAsayGulXn_5G1iBlGhdQ5BtZ_9
https://www.youtube.com/playlist?list=PLdhHip4wq4EPICgZZCjddQkvaqrRWgB-y
'''


ls_animated = ls_animated.split("\n")[1:-1]

done = []
for playlist_link in ls_animated:
    cmd = '/raid/infolab/bhavyakohli/executables/yt-dlp -o "/raid/infolab/bhavyakohli/parseq/data/vid_animations/%(title)s.%(ext)s" --no-overwrites --continue --ignore-errors ' + playlist_link
    ret = os.system(cmd)
    if ret == 2: 
        exit()
    done.append(playlist_link)
    with open("done_vid_animations.txt", "a+") as f:
        print(playlist_link, file=f)
    print(playlist_link, "done")
