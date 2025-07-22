from GSR import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Set domain parameters
x_min, x_max, t_min, t_max = -1., 1., 0., 1.
x_N, t_N = 100, 50

# Initialize Gaussian model
positions = get_grid_points(x_min, x_max, t_min, t_max, x_N, t_N).cpu().numpy()
model = GaussianSplattingFast(x_min, x_max, t_min, t_max, positions, dim=1)
model.load('output_pinns/gaussian_velocity_0.pt')

# Move model parameters to device
model.positions = model.positions.to(device)
model.scalings = model.scalings.to(device)
model.rotations = model.rotations.to(device)
model.values = model.values.to(device)

# Load PINN prediction
pinn_prediction = np.load('2D/prediction_of_pinn.npy')

# Load Burgers reference solution
def load_burgers_gt():
    npz_file = '2D/burgers_solution.npz'
    if os.path.exists(npz_file):
        data = np.load(npz_file)
        burgers_gt = data['solutions']
        print(f"Loaded Burgers reference solution: {npz_file}")
        return burgers_gt
    return None

burgers_gt = load_burgers_gt()

# Adjust dimensions to match prediction results
x_N, t_N = pinn_prediction.shape[0], pinn_prediction.shape[1]
print(f"Target dimensions: x_N={x_N}, t_N={t_N}")

if burgers_gt is not None:
    print(f"Original Burgers GT shape: {burgers_gt.shape}")
    
    # Handle 3D array by reducing dimensions
    if burgers_gt.ndim == 3:
        print("3D array detected, reducing dimensions...")
        if burgers_gt.shape[0] == 1:
            burgers_gt = burgers_gt[0]
        elif burgers_gt.shape[-1] == 1:
            burgers_gt = burgers_gt[:, :, 0]
        else:
            burgers_gt = burgers_gt[-1]
        print(f"Shape after dimension reduction: {burgers_gt.shape}")
    
    # Transpose Burgers reference solution
    print("Transposing Burgers reference solution...")
    burgers_gt = burgers_gt.T
    print(f"Shape after transpose: {burgers_gt.shape}")
    
    # Interpolate if dimensions don't match
    if burgers_gt.shape != (x_N, t_N):
        print("Dimensions mismatch, performing interpolation...")
        from scipy.interpolate import griddata
        
        if burgers_gt.ndim == 2:
            old_x_n, old_t_n = burgers_gt.shape
            
            # Create interpolation grids
            old_x = np.linspace(x_min, x_max, old_x_n)
            old_t = np.linspace(t_min, t_max, old_t_n)
            old_X, old_T = np.meshgrid(old_x, old_t, indexing='ij')
            
            new_x = np.linspace(x_min, x_max, x_N)
            new_t = np.linspace(t_min, t_max, t_N)
            new_X, new_T = np.meshgrid(new_x, new_t, indexing='ij')
            
            # Perform interpolation
            points = np.column_stack([old_X.ravel(), old_T.ravel()])
            values = burgers_gt.ravel()
            new_points = np.column_stack([new_X.ravel(), new_T.ravel()])
            
            burgers_gt_interpolated = griddata(points, values, new_points, method='linear')
            burgers_gt = burgers_gt_interpolated.reshape(x_N, t_N)
            
            print(f"Shape after interpolation: {burgers_gt.shape}")
        else:
            print("Abnormal Burgers GT dimensions, skipping comparison")
            burgers_gt = None

# Generate Gaussian prediction
x = torch.linspace(x_min, x_max, x_N)
t = torch.linspace(t_min, t_max, t_N)
X, T = torch.meshgrid(x, t, indexing='ij')
XT = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)

with torch.no_grad():
    gaussian_prediction = model(XT).cpu().numpy()[:, 0].reshape(x_N, t_N)

if burgers_gt is not None:
    # Verify shape consistency
    print(f"Final shape verification:")
    print(f"Burgers GT: {burgers_gt.shape}")
    print(f"Gaussian Prediction: {gaussian_prediction.shape}")
    print(f"PINN Prediction: {pinn_prediction.shape}")
    
    # Calculate errors (using Burgers reference as ground truth)
    error_gaussian_vs_burgers = np.abs(gaussian_prediction - burgers_gt)
    error_pinn_vs_burgers = np.abs(pinn_prediction - burgers_gt)
    
    # Visualization comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # First row: Three solutions
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
    
    # Second row: Error analysis
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
    
    # Error difference plot
    error_diff = error_pinn_vs_burgers - error_gaussian_vs_burgers
    im6 = axes[1, 2].imshow(error_diff, origin='lower', aspect='auto', 
                           extent=[t_min, t_max, x_min, x_max], cmap='RdBu_r')
    axes[1, 2].set_title('Error Difference\n(Red: Gaussian Better, Blue: PINN Better)')
    axes[1, 2].set_xlabel('Time t')
    axes[1, 2].set_ylabel('Space x')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('gaussian_vs_pinn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed error statistics
    print('=' * 80)
    print('Error Analysis with Burgers Reference as Ground Truth:')
    print('=' * 80)
    
    # Gaussian prediction vs Burgers reference
    mae_gaussian = np.mean(error_gaussian_vs_burgers)
    rmse_gaussian = np.sqrt(np.mean(error_gaussian_vs_burgers**2))
    max_error_gaussian = np.max(error_gaussian_vs_burgers)
    
    print('\n【Gaussian Splatting vs Burgers Reference】')
    print(f'MAE:  {mae_gaussian:.6f}')
    print(f'RMSE: {rmse_gaussian:.6f}')
    print(f'Max Error: {max_error_gaussian:.6f}')
    
    # PINN prediction vs Burgers reference
    mae_pinn = np.mean(error_pinn_vs_burgers)
    rmse_pinn = np.sqrt(np.mean(error_pinn_vs_burgers**2))
    max_error_pinn = np.max(error_pinn_vs_burgers)
    
    print('\n【PINN vs Burgers Reference】')
    print(f'MAE:  {mae_pinn:.6f}')
    print(f'RMSE: {rmse_pinn:.6f}')
    print(f'Max Error: {max_error_pinn:.6f}')
    
    # Comparison results
    print('\n' + '='*50)
    print('Accuracy Comparison Results:')
    print('='*50)
    
    mae_improvement = ((mae_pinn - mae_gaussian) / mae_pinn) * 100
    rmse_improvement = ((rmse_pinn - rmse_gaussian) / rmse_pinn) * 100
    
    print(f'\nMAE Improvement: {mae_improvement:.2f}%')
    print(f'RMSE Improvement: {rmse_improvement:.2f}%')
    
    if mae_gaussian < mae_pinn:
        print(f'\nConclusion: Gaussian Splatting is more accurate than PINN!')
        print(f'Gaussian MAE is {abs(mae_improvement):.2f}% lower than PINN')
    else:
        print(f'\nConclusion: PINN is more accurate than Gaussian Splatting')
        print(f'PINN MAE is {abs(mae_improvement):.2f}% lower than Gaussian')
    
    # Regional analysis
    print('\nRegional Analysis:')
    better_regions = np.sum(error_gaussian_vs_burgers < error_pinn_vs_burgers)
    total_regions = error_gaussian_vs_burgers.size
    accuracy_percentage = (better_regions / total_regions) * 100
    
    print(f'Gaussian is more accurate in {accuracy_percentage:.1f}% of regions')
    print(f'Better points: {better_regions}/{total_regions}')
    
    # Time evolution analysis
    print('\nTime Evolution Analysis:')
    gaussian_error_time = np.mean(error_gaussian_vs_burgers, axis=0)
    pinn_error_time = np.mean(error_pinn_vs_burgers, axis=0)
    
    better_time_steps = np.sum(gaussian_error_time < pinn_error_time)
    print(f'Gaussian is better in {better_time_steps}/{t_N} time steps')
    
    # Plot time evolution errors
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
    
    # Save comparison results
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
    print("Detailed comparison results saved to: gaussian_vs_pinn_results.npz")
    
else:
    print("Could not load Burgers reference solution, comparison analysis skipped")