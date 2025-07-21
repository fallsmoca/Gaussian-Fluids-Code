from GSR import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 假设你已经有 x_min, x_max, t_min, t_max, x_N, t_N
x_min, x_max, t_min, t_max = -1., 1., 0., 1.
x_N, t_N = 100, 50

positions = get_grid_points(x_min, x_max, t_min, t_max, x_N, t_N).cpu().numpy()
model = GaussianSplattingFast(x_min, x_max, t_min, t_max, positions, dim=1)
model.load('output_pinns/gaussian_velocity_0.pt')

# 加载后将所有参数移到正确设备
model.positions = model.positions.to(device)
model.scalings = model.scalings.to(device)
model.rotations = model.rotations.to(device)
model.values = model.values.to(device)

# 加载 PINN 预测解
pinn_prediction = np.load('2D/prediction.npy')  # PINN 预测解

# 加载 Burgers 参考解
def load_burgers_gt():
    """从当前目录加载 burgers_1d_solution.npz 文件"""
    
    # 优先查找指定的 npz 文件
    npz_file = '2D/burgers_1d_solution.npz'
    if os.path.exists(npz_file):
        print(f"找到 Burgers 参考解: {npz_file}")
        try:
            with np.load(npz_file) as data:
                print("NPZ 文件包含的键:", list(data.keys()))
                
                # 尝试不同的可能键名
                possible_keys = ['solution', 'u_field', 'velocity', 'data', 'arr_0', 'solutions']
                
                for key in possible_keys:
                    if key in data:
                        burgers_data = data[key]
                        print(f"使用键 '{key}' 加载数据")
                        print(f"数据形状: {burgers_data.shape}")
                        return burgers_data, npz_file
                
                # 如果没有找到已知键，使用第一个数组
                first_key = list(data.keys())[0]
                burgers_data = data[first_key]
                print(f"使用第一个键 '{first_key}' 加载数据")
                print(f"数据形状: {burgers_data.shape}")
                return burgers_data, npz_file
                
        except Exception as e:
            print(f"加载 {npz_file} 失败: {e}")
    
    print("未找到 Burgers 参考解文件")
    return None, None

burgers_gt, gt_path = load_burgers_gt()

# 调整维度以匹配预测结果
x_N, t_N = pinn_prediction.shape[0], pinn_prediction.shape[1]
print(f"目标维度: x_N={x_N}, t_N={t_N}")

if burgers_gt is not None:
    print(f"Burgers GT 原始形状: {burgers_gt.shape}")
    
    # 如果是3D数组，进行降维处理
    if burgers_gt.ndim == 3:
        print("检测到3D数组，进行降维处理...")
        if burgers_gt.shape[0] == 1:
            burgers_gt = burgers_gt[0]
        elif burgers_gt.shape[-1] == 1:
            burgers_gt = burgers_gt[:, :, 0]
        else:
            burgers_gt = burgers_gt[-1]  # 取最后一个时间步
        print(f"降维后形状: {burgers_gt.shape}")
    
    # 强制转置 Burgers 参考解
    print("对 Burgers 参考解执行转置...")
    burgers_gt = burgers_gt.T
    print(f"转置后形状: {burgers_gt.shape}")
    
    # 如果维度仍不匹配，进行插值调整
    if burgers_gt.shape != (x_N, t_N):
        print("维度不匹配，进行插值调整...")
        from scipy.interpolate import griddata
        
        # 获取原始维度
        if burgers_gt.ndim == 2:
            old_x_n, old_t_n = burgers_gt.shape
            
            # 创建插值网格
            old_x = np.linspace(x_min, x_max, old_x_n)
            old_t = np.linspace(t_min, t_max, old_t_n)
            old_X, old_T = np.meshgrid(old_x, old_t, indexing='ij')
            
            new_x = np.linspace(x_min, x_max, x_N)
            new_t = np.linspace(t_min, t_max, t_N)
            new_X, new_T = np.meshgrid(new_x, new_t, indexing='ij')
            
            # 插值调整
            points = np.column_stack([old_X.ravel(), old_T.ravel()])
            values = burgers_gt.ravel()
            new_points = np.column_stack([new_X.ravel(), new_T.ravel()])
            
            burgers_gt_interpolated = griddata(points, values, new_points, method='linear')
            burgers_gt = burgers_gt_interpolated.reshape(x_N, t_N)
            
            print(f"插值后 Burgers GT 形状: {burgers_gt.shape}")
        else:
            print("Burgers GT 维度异常，跳过对比")
            burgers_gt = None

# 生成 Gaussian 预测解
x = torch.linspace(x_min, x_max, x_N)
t = torch.linspace(t_min, t_max, t_N)
X, T = torch.meshgrid(x, t, indexing='ij')
XT = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)

with torch.no_grad():
    gaussian_prediction = model(XT).cpu().numpy()[:, 0].reshape(x_N, t_N)

if burgers_gt is not None:
    # 验证所有数组的形状一致性
    print(f"最终形状检查:")
    print(f"Burgers GT: {burgers_gt.shape}")
    print(f"Gaussian Prediction: {gaussian_prediction.shape}")
    print(f"PINN Prediction: {pinn_prediction.shape}")
    
    # 计算误差（以 Burgers 参考解为基准）
    error_gaussian_vs_burgers = np.abs(gaussian_prediction - burgers_gt)  # Gaussian 预测 vs Burgers
    error_pinn_vs_burgers = np.abs(pinn_prediction - burgers_gt)          # PINN 预测 vs Burgers
    
    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：三个解
    im1 = axes[0, 0].imshow(burgers_gt, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='RdBu_r')
    axes[0, 0].set_title('Burgers Reference Solution (Ground Truth)')
    axes[0, 0].set_xlabel('Time t')
    axes[0, 0].set_ylabel('Space x')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(gaussian_prediction, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='RdBu_r')
    axes[0, 1].set_title('Gaussian Splatting Prediction')
    axes[0, 1].set_xlabel('Time t')
    axes[0, 1].set_ylabel('Space x')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(pinn_prediction, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='RdBu_r')
    axes[0, 2].set_title('PINN Prediction')
    axes[0, 2].set_xlabel('Time t')
    axes[0, 2].set_ylabel('Space x')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 第二行：误差分析
    im4 = axes[1, 0].imshow(error_gaussian_vs_burgers, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='Reds')
    axes[1, 0].set_title('Error: Gaussian vs Burgers Reference')
    axes[1, 0].set_xlabel('Time t')
    axes[1, 0].set_ylabel('Space x')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(error_pinn_vs_burgers, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='Reds')
    axes[1, 1].set_title('Error: PINN vs Burgers Reference')
    axes[1, 1].set_xlabel('Time t')
    axes[1, 1].set_ylabel('Space x')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 误差对比图
    error_diff = error_pinn_vs_burgers - error_gaussian_vs_burgers  # 正值表示 Gaussian 更准确
    im6 = axes[1, 2].imshow(error_diff, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='RdBu_r')
    axes[1, 2].set_title('Error Difference\n(Red: Gaussian Better, Blue: PINN Better)')
    axes[1, 2].set_xlabel('Time t')
    axes[1, 2].set_ylabel('Space x')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('gaussian_vs_pinn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 详细误差统计
    print('=' * 80)
    print('以 Burgers 参考解为基准的误差分析:')
    print('=' * 80)
    
    # Gaussian 预测 vs Burgers 参考解
    mae_gaussian = np.mean(error_gaussian_vs_burgers)
    rmse_gaussian = np.sqrt(np.mean(error_gaussian_vs_burgers**2))
    max_error_gaussian = np.max(error_gaussian_vs_burgers)
    
    print('\n【Gaussian Splatting 预测 vs Burgers 参考解】')
    print(f'MAE:  {mae_gaussian:.6f}')
    print(f'RMSE: {rmse_gaussian:.6f}')
    print(f'Max Error: {max_error_gaussian:.6f}')
    
    # PINN 预测 vs Burgers 参考解
    mae_pinn = np.mean(error_pinn_vs_burgers)
    rmse_pinn = np.sqrt(np.mean(error_pinn_vs_burgers**2))
    max_error_pinn = np.max(error_pinn_vs_burgers)
    
    print('\n【PINN 预测 vs Burgers 参考解】')
    print(f'MAE:  {mae_pinn:.6f}')
    print(f'RMSE: {rmse_pinn:.6f}')
    print(f'Max Error: {max_error_pinn:.6f}')
    
    # 比较结果
    print('\n' + '='*50)
    print('🏆 准确性对比结果:')
    print('='*50)
    
    mae_improvement = ((mae_pinn - mae_gaussian) / mae_pinn) * 100
    rmse_improvement = ((rmse_pinn - rmse_gaussian) / rmse_pinn) * 100
    
    print(f'\nMAE 改善: {mae_improvement:.2f}%')
    print(f'RMSE 改善: {rmse_improvement:.2f}%')
    
    if mae_gaussian < mae_pinn:
        print('\n✅ 结论: Gaussian Splatting 预测比 PINN 预测更接近 Burgers 参考解!')
        print(f'   Gaussian 的 MAE 比 PINN 低 {abs(mae_improvement):.2f}%')
    else:
        print('\n❌ 结论: PINN 预测比 Gaussian Splatting 预测更接近 Burgers 参考解')
        print(f'   PINN 的 MAE 比 Gaussian 低 {abs(mae_improvement):.2f}%')
    
    # 区域性分析
    print('\n📊 区域性分析:')
    better_regions = np.sum(error_gaussian_vs_burgers < error_pinn_vs_burgers)
    total_regions = error_gaussian_vs_burgers.size
    accuracy_percentage = (better_regions / total_regions) * 100
    
    print(f'Gaussian 预测在 {accuracy_percentage:.1f}% 的区域比 PINN 预测更准确')
    print(f'Gaussian 更准确的点数: {better_regions}/{total_regions}')
    
    # 时间演化分析
    print('\n⏰ 时间演化分析:')
    gaussian_error_time = np.mean(error_gaussian_vs_burgers, axis=0)  # 每个时间步的平均误差
    pinn_error_time = np.mean(error_pinn_vs_burgers, axis=0)
    
    better_time_steps = np.sum(gaussian_error_time < pinn_error_time)
    print(f'Gaussian 预测在 {better_time_steps}/{t_N} 个时间步比 PINN 预测更准确')
    
    # 绘制时间演化误差图
    plt.figure(figsize=(12, 6))
    t_array = np.linspace(t_min, t_max, t_N)
    
    plt.subplot(1, 2, 1)
    plt.plot(t_array, gaussian_error_time, 'r-', label='Gaussian Error', linewidth=2)
    plt.plot(t_array, pinn_error_time, 'b-', label='PINN Error', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    error_ratio = gaussian_error_time / pinn_error_time
    plt.plot(t_array, error_ratio, 'g-', linewidth=2)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal Error')
    plt.xlabel('Time t')
    plt.ylabel('Gaussian Error / PINN Error')
    plt.title('Error Ratio (< 1 means Gaussian Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_time_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('=' * 80)
    
    # 保存对比结果
    comparison_results = {
        'burgers_reference': burgers_gt,
        'gaussian_prediction': gaussian_prediction,
        'pinn_prediction': pinn_prediction,
        'error_gaussian_vs_burgers': error_gaussian_vs_burgers,
        'error_pinn_vs_burgers': error_pinn_vs_burgers,
        'metrics': {
            'mae_gaussian': mae_gaussian,
            'rmse_gaussian': rmse_gaussian,
            'max_error_gaussian': max_error_gaussian,
            'mae_pinn': mae_pinn,
            'rmse_pinn': rmse_pinn,
            'max_error_pinn': max_error_pinn,
            'mae_improvement': mae_improvement,
            'rmse_improvement': rmse_improvement,
            'accuracy_percentage': accuracy_percentage
        }
    }
    
    np.savez('gaussian_vs_pinn_results.npz', **comparison_results)
    print("详细对比结果已保存到: gaussian_vs_pinn_results.npz")
    
else:
    print("未能加载Burgers参考解，无法进行对比分析")