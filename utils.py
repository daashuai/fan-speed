import numpy as np

def calculate_energy(fan_speeds, time_step=1.0):
    """
    计算总能耗。
    
    Args:
        fan_speeds (list or np.ndarray): 风扇速度记录，单位为 RPM。
        time_step (float): 时间步的长度，单位为秒。
    
    Returns:
        float: 总能耗。
    """
    fan_speeds = np.array(fan_speeds)
    energy = np.sum(fan_speeds**2) * time_step
    return energy


def calculate_speed_smoothness(fan_speeds):
    """
    计算控制平稳性的均方差。
    
    Args:
        fan_speeds (list or np.ndarray): 风扇速度记录，单位为 RPM。
    
    Returns:
        float: 控制平稳性（速度变化的均方差）。
    """
    fan_speeds = np.array(fan_speeds)
    changes = np.diff(fan_speeds)  # 计算速度变化
    smoothness = np.mean(changes**2)
    return smoothness

def calculate_max_change(fan_speeds):
    """
    计算风扇速度的最大变化量。
    
    Args:
        fan_speeds (list or np.ndarray): 风扇速度记录，单位为 RPM。
    
    Returns:
        float: 最大变化量。
    """
    fan_speeds = np.array(fan_speeds)
    changes = np.diff(fan_speeds)  # 计算速度变化
    max_change = np.max(np.abs(changes))
    return max_change

def calculate_stabilization_time(fan_speeds, target_speed, target_tolerance, time_step=1.0):
    """
    计算达到稳定状态的时间。
    
    Args:
        fan_speeds (list or np.ndarray): 风扇速度记录，单位为 RPM。
        target_speed (float): 目标风扇速度。
        target_tolerance (float): 容忍范围，单位为 RPM。
        time_step (float): 时间步的长度，单位为秒。
    
    Returns:
        float or None: 稳定时间（秒），若始终未稳定，则返回 None。
    """
    fan_speeds = np.array(fan_speeds)
    within_tolerance = np.abs(fan_speeds - target_speed) <= target_tolerance
    for i in range(len(fan_speeds)):
        if np.all(within_tolerance[i:]):  # 检查从时间步 i 到最后是否都在容忍范围内
            return i * time_step
    return None

def calculate_speed_deviation(fan_speeds, target_speed):
    """
    计算风扇速度偏离目标速度的均方误差。
    
    Args:
        fan_speeds (list or np.ndarray): 风扇速度记录，单位为 RPM。
        target_speed (float): 目标风扇速度。
    
    Returns:
        float: 偏离目标速度的均方误差。
    """
    fan_speeds = np.array(fan_speeds)
    deviation = np.mean((fan_speeds - target_speed)**2)
    return deviation

def calculate_temp_deviation(temps, target_temp=40):
    temps = np.array(temps)
    deviation = np.mean((temps - target_temp)**2)
    return deviation


def plot_speed_temp(writer, epoch, speeds, temps):
    # view_start = 100
    view_start = 20
    test_length = len(speeds)
    if test_length <= view_start+1:
        return
    speed_max = max(speeds)
    speed_min = min(speeds[view_start:])
    temp_max = max(temps)
    temp_min = min(temps[view_start:])
    # test_length = self.max_steps 
    
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(15,8))
    ax2 = ax1.twinx()
    ax1.plot(range(0, test_length), speeds, 'green', linewidth=3, label='Fan Speed')
    ax2.plot(range(0, test_length), temps, 'red', linewidth=3, label='Temperature')
    ax1.set_ylabel("Fan Speed")
    ax2.set_ylabel("Temperature")

    ax1.annotate(str(speed_max), xy=(speeds.index(speed_max), speed_max),
         xytext=(speeds.index(speed_max) + 100, speed_max + 350), arrowprops=dict(facecolor='black', shrink=1))
    ax1.annotate(str(speed_min), xy=(speeds.index(speed_min), speed_min),
         xytext=(speeds.index(speed_min) + 100, speed_min - 500), arrowprops=dict(facecolor='black', shrink=1))

    ax2.annotate(str(temp_max), xy=(temps.index(temp_max), temp_max),
         xytext=(temps.index(temp_max) - 100, temp_max + 2), arrowprops=dict(facecolor='orange', shrink=1))
    ax2.annotate(str(temp_min), xy=(temps.index(temp_min), temp_min),
         xytext=(temps.index(temp_min) + 50, temp_min - 2), arrowprops=dict(facecolor='orange', shrink=1))
    fig.legend(loc='upper right')

    # fig.savefig(experiment_dir + 'image/' + 'sac-' + time.strftime('%Y-%m-%d %H:%M', time.localtime()) + '.jpg')
    # 将图像保存到内存（而不是文件）
    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    # 转为 NumPy 数组
    image = plt.imread(buffer, format="png")
    writer.add_image("Plot/"+str(epoch), image, global_step=epoch, dataformats="HWC")
    buffer.close()
    writer.close()


    # terminal = sys.stdout
    # sys.stdout = open('./image/speed-temp.log','a')
    # print("speeds:")
    # print(speeds)
    # print("temps:")
    # print(temps)
    # sys.stdout = terminal


