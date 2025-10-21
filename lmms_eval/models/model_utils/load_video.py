import av
from av.codec.context import CodecContext
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def read_video_pyav(video_path, num_frm=8):
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = record_video_length_stream(container, indices)
        except:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = [frames[i] for i in indices]
    else:
        container = av.open(video_path)
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        sampled_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        frames = [frames[i] for i in indices]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_frame_indices(total_frames, original_fps, target_fps, num_frm):
    # target_fps = original_fps  # hardcode, should remove after debug
    sample_fps = max(round(original_fps / target_fps), 1)
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        # If we have fewer frames than num_frm, just return all the frames
        return frame_idx
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def add_timestamp_to_frame(frame, start_sec, end_sec, font_size=40):
    # 计算时间戳区域高度（基于原图像高度）
    timestamp_height = int(frame.height * 0.04)  # 原图像高度的10%
    
    # 创建新的画布（高度增加时间戳区域）
    new_height = frame.height + timestamp_height
    new_frame = Image.new('RGB', (frame.width, new_height), color=(255, 255, 255))
    
    # 将原图像粘贴到新画布的下半部分
    new_frame.paste(frame, (0, timestamp_height))
    
    # 在新画布的上半部分添加时间戳
    draw = ImageDraw.Draw(new_frame)
    font_size = int(frame.height * 0.04)
    #font = ImageFont.load_default(font_size)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    
    text = f"{sec2hms(start_sec)}-{sec2hms(end_sec)}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 时间戳居中显示在上方区域
    x = (frame.width - text_w) // 2
    y = (timestamp_height - text_h - 3) // 2
    
    # 修复矩形绘制：使用 RGB 颜色，不透明
    draw.rectangle([x-15, y-6, x+text_w+15, y+text_h+6], fill=(50, 50, 50))  # 移除了透明度
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return new_frame
    
def sec2hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_video_pyav2(video_path, num_frm=8, target_fps=1, add_ts=False, threads=8):
    container = av.open(video_path)
    stream = container.streams.video[0]

    stream.thread_type = 'AUTO'
    stream.codec_context.thread_count = threads

    original_fps = stream.average_rate
    total_frames = stream.frames

    if "webm" not in video_path and "mkv" not in video_path:
        try:
            indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
            frames = record_video_length_stream(container, indices)
        except:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
            frames = [frames[i] for i in indices]
    else:
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
        frames = [frames[i] for i in indices]

    frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    total_time = stream.frames / original_fps
    time_interval = round(total_time/len(indices),3)
    start_sec, end_sec = 0,0
    frames_with_ts = []
    
    if add_ts:
        for i, frame in enumerate(frames):
            start_sec = i * time_interval
            end_sec = start_sec + time_interval
            frame_with_ts = add_timestamp_to_frame(frame, start_sec, end_sec)
            frames_with_ts.append(frame_with_ts)
        frames = frames_with_ts
    
    total_time = stream.frames / original_fps
    video_info_string = f"Time: {round(total_time, 2)}s; Time interval between frame {round(total_time/len(indices), 3)}s; video tokens:"

    return frames, video_info_string