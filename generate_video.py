import cv2
import os

def sort_frames_and_save_as_video(frames_dir, output_video_path):
    # 获取帧文件名列表并按照文件名排序
    frame_files = sorted(os.listdir(frames_dir))

    # 读取第一帧以获取帧的尺寸
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # 逐帧读取并写入视频
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()

    print(f"Video saved as {output_video_path}")

# 设置帧文件夹路径和输出视频路径
frames_directory = "path/to/frames_directory"
output_video_path = "path/to/output_video.mp4"

# 调用函数将帧合成为视频
sort_frames_and_save_as_video(frames_directory, output_video_path)
