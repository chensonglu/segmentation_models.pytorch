import os
import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


def convert_to_color_image(matrix):
    """
    将单通道的0,1,2矩阵转换为彩色图像。
    0 -> 黑色, 1 -> 红色, 2 -> 蓝色
    """
    color_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

    # 映射规则：0 -> 黑色, 1 -> 红色, 2 -> 蓝色
    color_image[matrix == 0] = [0, 0, 0]   # 黑色
    color_image[matrix == 1] = [0, 0, 255] # 红色
    color_image[matrix == 2] = [255, 0, 0] # 蓝色

    return color_image


CLASSES = [
    "good",
    "bad",
    "unlabelled",
]
arch = 'FPN'
encoder_name = 'resnet18'
dir_name = 'resnet18'
in_channels = 3
out_classes = len(CLASSES)

model = CamVidModel.load_from_checkpoint(os.path.join("lightning_logs_leaf", arch+"_"+dir_name, 'checkpoints/epoch=49-step=50.ckpt'), **{'arch': arch, 'encoder_name': encoder_name, 'in_channels': in_channels, 'out_classes': out_classes})
model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")
model.to(device)

pid = os.getpid()
print(f"The PID of the current process is: {pid}")

# 定义RTSP视频流的URL地址或视频文件路径
rtsp_url = "rtsp://admin:123456@192.168.10.154:554/stream_0"

# 创建VideoCapture对象，用于捕获视频流
video_capture = cv2.VideoCapture(rtsp_url)

# 检查是否成功打开视频流
if not video_capture.isOpened():
    print("Error: Could not open video stream or file.")
    exit()

# 获取视频流的属性
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(fps)

# 定义输出视频文件的路径和格式
output_video_file = "output_video.avi"

# 创建VideoWriter对象，用于保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 根据需要调整编码器

video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

# # 创建保存图像的目录
# image_folder = "saved_images"
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# 读取并保存视频流和图像
frame_count = 0
while True:
    ret, image = video_capture.read()
    if not ret:
        print("Error: Could not read frame from video stream.")
        break

    # # 保存图像
    # image_filename = os.path.join(image_folder, f"frame_{frame_count:04d}.png")
    # cv2.imwrite(image_filename, image)
    # frame_count += 1

    # 检查图片是否成功加载
    if image is None:
        print("Error: Image not found or unable to read.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        image = image.to(device)
        assert image.ndim == 4

        logits_mask = model(image)

        assert (
            logits_mask.shape[1] == model.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        pred_mask = np.squeeze(pred_mask.cpu().numpy(), axis=0)
        print(pred_mask.shape)
        pred_mask = cv2.resize(pred_mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        pred_mask_color = convert_to_color_image(pred_mask)

        print(pred_mask_color.shape)

        video_writer.write(pred_mask_color)

        # count_ones = np.count_nonzero(pred_mask.cpu().numpy() == 1)
        # print(count_ones)




# import numpy as np
# import cv2

# def convert_to_color_image(matrix):
#     """
#     将单通道的0,1,2矩阵转换为彩色图像。
#     0 -> 黑色, 1 -> 红色, 2 -> 蓝色
#     """
#     # 去掉冗余的维度
#     matrix = matrix.squeeze()  # 使matrix形状从(320, 320, 1)变为(320, 320)
    
#     # 创建一个彩色图像，320x320的尺寸，3通道 (RGB)
#     color_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

#     # 映射规则：0 -> 黑色, 1 -> 红色, 2 -> 蓝色
#     color_image[matrix == 0] = [0, 0, 0]   # 黑色
#     color_image[matrix == 1] = [0, 0, 255] # 红色
#     color_image[matrix == 2] = [255, 0, 0] # 蓝色

#     return color_image

# def save_video(matrices, video_filename, frame_rate=30, video_size=(320, 320)):
#     """
#     将多个矩阵保存为视频文件。
    
#     matrices: 一个形状为 (N, 320, 320, 1) 的数组，包含N帧，每帧是320x320x1的矩阵
#     video_filename: 保存的视频文件名
#     frame_rate: 视频帧率，默认为30fps
#     video_size: 视频尺寸，默认为320x320
#     """
#     # 设置视频编码格式和输出文件
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_filename, fourcc, frame_rate, video_size)
    
#     for matrix in matrices:
#         # 转换为彩色图像
#         color_image = convert_to_color_image(matrix)
#         # 将图像写入视频
#         out.write(color_image)

#     # 释放视频写入对象
#     out.release()

# # 假设 matrices 是一个形状为 (N, 320, 320, 1) 的数组，其中N是矩阵数量
# # 每个矩阵是 320x320x1，矩阵值为 0, 1, 2 之一
# # 示例：
# N = 100  # 假设有100帧
# matrices = np.random.randint(0, 3, (N, 320, 320, 1))  # 随机生成一些示例矩阵

# # 将这些矩阵保存到一个视频文件中
# save_video(matrices, 'output_video.avi', frame_rate=30)
