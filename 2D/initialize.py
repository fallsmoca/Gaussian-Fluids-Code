import torch
import torch.nn.functional as F
import numpy as np
import time
from GSR import *
import matplotlib.pyplot as plt


def fit_velocity_with_gradient(gaussian_velocity: GaussianSplattingFast, data_generator, data_boundary_generator, data_initial_generator, batch_size=256, max_epoch=10000, verbose=1):
    total_losses, pde_losses, bnd_losses, init_losses = [], [], [], []

    total_start_time = time.time()
    st_time = time.time()
    gaussian_velocity.initialize_optimizers()

    for epoch in range(max_epoch):
        # calculate difference between gaussian_splatting and reference_field
        data = data_generator(batch_size)
        data_boundary = data_boundary_generator(batch_size)
        data_initial = data_initial_generator(batch_size)
        pde_loss = gaussian_velocity.get_losses(data, weight=1., weight_boundary=0., weight_initial=0.)[1].mean()
        boundary_loss = gaussian_velocity.get_losses(data_boundary, weight=0., weight_boundary=1., weight_initial=0.)[2].mean()
        initial_loss = gaussian_velocity.get_losses(data_initial, weight=0., weight_boundary=0., weight_initial=1.)[3].mean()
        loss = pde_loss + boundary_loss + initial_loss

        total_losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        bnd_losses.append(boundary_loss.item())
        init_losses.append(initial_loss.item())

        gaussian_velocity.step(loss)

        if verbose and epoch % 100 == 99:
            with torch.no_grad():
                en_time = time.time()
                elapsed_100_epochs = en_time - st_time
                print(f'Epoch {epoch + 1}/{max_epoch}, loss:{loss}, pde:{pde_loss}, bnd:{boundary_loss}, init:{initial_loss}')
                print(f'Last 100 epochs time: {elapsed_100_epochs:.2f}s')
                st_time = time.time()
                #show_field(gaussian_velocity, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0, x_N=200, t_N=100, save_filename=os.path.join(cmd_args.dir, f'prediction_xt_{epoch + 1}.png'))

    # 计算总训练时间
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    avg_100_epochs_time = total_training_time / (max_epoch / 100)

    print("="*50)
    print(f"Training completed!")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.1f} minutes)")
    print(f"Average time per 100 epochs: {avg_100_epochs_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/max_epoch:.3f} seconds")
    print("="*50)

    plt.figure(figsize=(6,4))
    plt.plot(total_losses, label='total loss')
    plt.plot(pde_losses, label='PDE loss')
    plt.plot(bnd_losses, label='boundary loss')
    plt.plot(init_losses, label='initial loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log') 
    plt.legend()
    plt.title('Training Loss Curve')
    plt.show()

    lr_ratio = 1.201956
    gaussian_velocity.set_lr(positions_lr=1e-5 * 10., scalings_lr=1e-5, rotations_lr=1e-5 * lr_ratio, values_lr=1e-5 * 10.)
    gaussian_velocity.initialize_optimizers(patience=50)
    gaussian_velocity.positions_scheduler.factor = .9
    gaussian_velocity.scalings_scheduler.factor = .9
    gaussian_velocity.rotations_scheduler.factor = .9
    gaussian_velocity.values_scheduler.factor = .9


def SimulationInitialize():
	x_min, x_max, y_min, y_max = -1.,1.,0.,1.
	
	# ref_velocity = eval(cmd_args.init_cond)
	# def velocity_field(x):
	# 	return ref_velocity(x, False)
	# def velocity_gradient(x):
	# 	return ref_velocity(x, True)
	# def vorticity_field(x):
	# 	g = velocity_gradient(x)
	# 	if len(x.shape) == 1:
	# 		return g[1, 0] - g[0, 1]
	# 	return g[:, 1, 0] - g[:, 0, 1]
	# def divergence_field(x):
	# 	g = velocity_gradient(x)
	# 	if len(x.shape) == 1:
	# 		return g[0, 0] + g[1, 1]
	# 	return g[:, 0, 0] + g[:, 1, 1]
	
	# show_field(velocity_field, x_min, x_max, y_min, y_max, dim=2, x_N=30, y_N=30, save_filename=os.path.join(cmd_args.dir, 'refvelocity.png'))
	# show_field(vorticity_field, x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'refvorticity.png'))
	# show_field(divergence_field, x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'refdivergence.png'))
	
	x_N, y_N = 60,30
	gaussian_velocity = GaussianSplattingFast(x_min, x_max, y_min, y_max, get_grid_points(x_min, x_max, y_min, y_max, x_N, y_N).cpu().numpy(), dim=1)
	print(f'Particle count: {gaussian_velocity.N} ({x_N} x {y_N})')
	
	def default_generator(n):
		return (torch.rand_like(gaussian_velocity.positions, device=device) * torch.tensor([x_max - x_min, y_max - y_min], device=device) + torch.tensor([x_min, y_min], device=device))
	def boundary_data_generator(n):
		n_half = n // 2
		x_coords = torch.cat([
			torch.full((n_half, 1), x_min, device=device),
			torch.full((n - n_half, 1), x_max, device=device)
		])
		y_coords = torch.rand(n, 1, device=device) * (y_max - y_min) + y_min
		data = torch.cat([x_coords, y_coords], dim=1)
		return data 
	def initial_data_generator(n):
		x = torch.rand(n, device=device) * (x_max - x_min) + x_min
		y = torch.zeros(n, device=device)
		data = torch.stack([x, y], dim=1)
		return data
	#positions_lr=1.6e-3, scalings_lr=5e-2, rotations_lr=5e-2, values_lr=5e-3
	gaussian_velocity.set_lr(positions_lr=5.e-4, scalings_lr=1.e-2, rotations_lr=1.e-2, values_lr=1.e-2)
	fit_velocity_with_gradient(gaussian_velocity, default_generator, boundary_data_generator, initial_data_generator, max_epoch=5000)
	gaussian_velocity.save(os.path.join(cmd_args.dir, 'gaussian_velocity_0.pt'))
	show_field(gaussian_velocity, x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0, x_N=200, t_N=100, save_filename=os.path.join(cmd_args.dir, 'prediction_xt.png'))
	# def vorticity_gaussian(x):
	# 	g = gaussian_velocity.gradient(x)
	# 	return g[:, 1, 0] - g[:, 0, 1]
	# def divergence_gaussian(x):
	# 	g = gaussian_velocity.gradient(x)
	# 	return g[:, 0, 0] + g[:, 1, 1]
	# show_field(original_gradient(vorticity_gaussian), x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'vorticity_0.png'))
	# show_field(original_gradient(divergence_gaussian), x_min, x_max, y_min, y_max, x_N=x_Nvis, y_N=y_Nvis, save_filename=os.path.join(cmd_args.dir, 'divergence_0.png'))


if __name__ == '__main__':
    SimulationInitialize()