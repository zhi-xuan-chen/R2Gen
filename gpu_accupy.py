
# import torch

# def occupy_gpu_memory(device, memory_size):
#     try:
#         # 检查指定的显卡是否可用
#         if torch.cuda.is_available() and device < torch.cuda.device_count():
#             # 设置当前设备
#             torch.cuda.set_device(device)
#             # 创建一个需要占用显存的张量
#             tensor_size = (memory_size * 1024 * 1024) // 4  # 一个 float32 数值占用 4 字节
#             tensor = torch.empty(tensor_size).cuda()
#             print(f"在 GPU {device} 上占用了 {memory_size}MB 的显存。")
#             # 等待停止程序
#             input("按任意键停止程序以释放显存...")
#         else:
#             print("指定的显卡不可用或不存在。")
#     except RuntimeError as e:
#         print("发生错误:", e)

# # 指定要占用显存的显卡索引和显存大小（以MB为单位）
# gpu_index = 0  # 在第一个显卡上占用显存
# memory_size = 16000  # 占用500MB的显存

# # 启动占用显存的程序
# occupy_gpu_memory(gpu_index, memory_size)

import torch

def occupy_gpu_memory(device, memory_size, mode='compute'):
    try:
        # 检查指定的显卡是否可用
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            # 设置当前设备
            torch.cuda.set_device(device)
            # 创建一个需要占用显存的张量
            tensor_size = (memory_size * 1024 * 1024) // 4  # 一个 float32 数值占用 4 字节
            tensor = torch.empty(tensor_size).cuda()
            print(f"在 GPU {device} 上占用了 {memory_size}MB 的显存。")
            
            if mode == 'compute':
                # 等待停止程序
                print("停止程序以释放显存...")
                # 持续进行计算
                while True:
                    matrix_size = 1000
                    matrix_a = torch.randn(matrix_size, matrix_size).cuda()
                    matrix_b = torch.randn(matrix_size, matrix_size).cuda()
                    result = torch.matmul(matrix_a, matrix_b)
                    # 在这里可以添加其他计算任务
            elif mode == 'memory':
                # 等待停止程序
                print("停止程序以释放显存...")
        else:
            print("指定的显卡不可用或不存在。")
    except RuntimeError as e:
        print("发生错误:", e)

# 指定要占用显存的显卡索引和显存大小（以MB为单位）
gpu_index = 0  # 在第一个显卡上占用显存
memory_size = 8000  # 占用16000MB的显存
mode = 'compute'  # 计算模式

# 启动占用显存的程序
occupy_gpu_memory(gpu_index, memory_size, mode)