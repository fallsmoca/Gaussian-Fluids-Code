import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
import taichi as ti
import taichi.math as tm
import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--dir', type=str, default='output_fast')
	parser.add_argument('--start_frame', type=int, default=0)
	parser.add_argument('--init_cond', type=str, default='taylor_vortex')
	parser.add_argument('--dt', type=float, default=.01)
	parser.add_argument('--last_time', type=float, default=10.)
	return parser.parse_args()
cmd_args = parse_args()
os.makedirs(cmd_args.dir, exist_ok=True)

torch.manual_seed(42)
if cmd_args.device != 'cpu':
	os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.device
	torch.cuda.manual_seed_all(42)
device = torch.device('cpu' if cmd_args.device == 'cpu' else 'cuda')
ti.init(arch=ti.cpu if cmd_args.device == 'cpu' else ti.cuda)

TiArr = ti.types.ndarray()


class GaussianSplatting:
	def __init__(self, positions, dim):
		self.N, self.dim = positions.shape[0], dim
		
		self.positions = torch.tensor(positions, dtype=torch.float, requires_grad=True, device=device)
		self.scalings = torch.zeros((self.N, 2), requires_grad=True, device=device) # it's actually scalings reverse
		self.rotations = torch.zeros(self.N, requires_grad=True, device=device)
		self.values = torch.zeros((self.N, dim), requires_grad=True, device=device)
	
	def set_lr(self, positions_lr, scalings_lr, rotations_lr, values_lr):
		self.positions_lr = positions_lr
		self.scalings_lr = scalings_lr
		self.rotations_lr = rotations_lr
		self.values_lr = values_lr
	
	def initialize_optimizers(self, patience=50):
		self.positions_optimizer = torch.optim.Adam([self.positions], lr=self.positions_lr)
		self.positions_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.positions_optimizer, factor=.9, patience=patience)
		self.scalings_optimizer = torch.optim.Adam([self.scalings], lr=self.scalings_lr)
		self.scalings_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.scalings_optimizer, factor=.9, patience=patience)
		self.rotations_optimizer = torch.optim.Adam([self.rotations], lr=self.rotations_lr)
		self.rotations_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.rotations_optimizer, factor=.9, patience=patience)
		self.values_optimizer = torch.optim.Adam([self.values], lr=self.values_lr)
		self.values_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.values_optimizer, factor=.9, patience=patience)
		
		self.optimizers = [
			self.positions_optimizer,
			self.scalings_optimizer,
			self.rotations_optimizer,
			self.values_optimizer,
		]
		self.schedulers = [
			self.positions_scheduler,
			self.scalings_scheduler,
			self.rotations_scheduler,
			self.values_scheduler,
		]
	
	def parameters(self):
		return {
			'positions': self.positions,
			'scalings': self.scalings,
			'rotations': self.rotations,
			'values': self.values
		}
	
	def save(self, filename):
		torch.save(self.parameters(), filename)
	
	def load(self, filename):
		parameters_dict = torch.load(filename)
		self.positions = parameters_dict['positions']
		self.scalings = parameters_dict['scalings']
		self.rotations = parameters_dict['rotations']
		self.values = parameters_dict['values']
		self.N = self.positions.shape[0]
		self.dim = self.values.shape[1]
	
	def get_scaling_matrices(self):
		return torch.diag_embed(torch.exp(self.scalings)) # re-parameterize scalings
	
	def get_rotation_matrices(self):
		R = torch.zeros((self.rotations.shape[0], 2, 2), device=device)
		rotations = self.rotations[:] # torch.sigmoid(self.rotations[:]) * 2. * torch.pi # re-parameterize rotations
		R[:, 0, 0] = R[:, 1, 1] = torch.cos(rotations)
		R[:, 0, 1] = -torch.sin(rotations)
		R[:, 1, 0] = torch.sin(rotations)
		return R
	
	def get_variances(self):
		S = self.get_scaling_matrices()
		R = self.get_rotation_matrices()
		A = R @ S
		return A @ A.permute(0, 2, 1)
	
	def forward_single(self, x):
		mu, sigma_inv = self.positions, self.get_variances()
		per_splatting_value = self.values * torch.exp(-.5 * (x - mu)[:, None, :] @ sigma_inv @ (x - mu)[:, :, None]).squeeze(2)
		return per_splatting_value.sum(axis=0)
	
	def __call__(self, x):
		if len(x.shape) == 1:
			return self.forward_single(x)
		mu, sigma_inv = self.positions, self.get_variances()
		positions_differences = x[:, None, :] - mu[None, :, :]
		per_splatting_values = self.values * torch.exp(-.5 * positions_differences[:, :, None, :] @ sigma_inv @ positions_differences[:, :, :, None]).squeeze(3)
		return per_splatting_values.sum(axis=1)
	
	def gradient_single(self, x, need_val=False):
		'''
		Compute the gradiant w.r.t. the single input.
		Returns a tensor with shape (self.dim, 2).
		'''
		mu, sigma_inv = self.positions, self.get_variances()
		per_splatting_value = self.values * torch.exp(-.5 * (x - mu)[:, None, :] @ sigma_inv @ (x - mu)[:, :, None]).squeeze(2)
		y = per_splatting_value.sum(axis=0)
		grad = -(per_splatting_value[:, :, None] @ (sigma_inv @ (x - mu)[:, :, None]).transpose(-1, -2)).sum(axis=0)
		return (grad, y) if need_val else grad
	
	def gradient(self, x, need_val=False):
		'''
		Compute the gradient w.r.t. the input.
		Returns a tensor with shpae (x.shape[0], self.dim, 2).
		'''
		if len(x.shape) == 1:
			return self.gradient_single(x, need_val)
		
		mu, sigma_inv = self.positions, self.get_variances()
		positions_differences = x[:, None, :] - mu[None, :, :]
		per_splatting_values = self.values * torch.exp(-.5 * positions_differences[:, :, None, :] @ sigma_inv @ positions_differences[:, :, :, None]).squeeze(3)
		y = per_splatting_values.sum(axis=1)
		grad = -(per_splatting_values[:, :, :, None] @ (sigma_inv @ positions_differences[:, :, :, None]).transpose(-1, -2)).sum(axis=1)
		return (grad, y) if need_val else grad
	
	def freeze(self):
		self.positions.requires_grad_(False)
		self.scalings.requires_grad_(False)
		self.rotations.requires_grad_(False)
		self.values.requires_grad_(False)
	
	def unfreeze(self):
		self.positions.requires_grad_()
		self.scalings.requires_grad_()
		self.rotations.requires_grad_()
		self.values.requires_grad_()
	
	def zero_grad(self):
		for o in self.optimizers:
			o.zero_grad()
	
	def step(self, metrics):
		for o in self.optimizers:
			o.step()
		for s in self.schedulers:
			s.step(metrics)


@ti.data_oriented
class GaussianSplattingFast(GaussianSplatting):
		def __init__(self, x_min, x_max, y_min, y_max, positions, min_grid_scale=None, clamp_threshold=1e-3, dim=1, load_file=None):
			super().__init__(positions, dim)
			
			# 添加PDE loss三个部分的存储
			self.pde_loss_parts = {
				'A': 0.0,  # Σ_i v_i ∂G_i(x)/∂t
				'B': 0.0,  # Σ_i (G_i(x)-c)v_i
				'C': 0.0   # Σ_i v_i ∂G_i(x)/∂x
			}
			
			if load_file is None:
				self.min_grid_scale = ((x_max - x_min) * (y_max - y_min) / self.N) ** .5 * 3. if (min_grid_scale is None) else min_grid_scale
				self.clamp_threshold = clamp_threshold
				self.x_min, self.x_max, self.y_min, self.y_max = x_min - self.min_grid_scale, x_max + self.min_grid_scale, y_min - self.min_grid_scale, y_max + self.min_grid_scale
				with torch.no_grad():
					self.scalings += .5 * np.log(-2. * np.log(self.clamp_threshold)) - np.log(self.min_grid_scale)
				self.create_grid_data()
				self.zero_grad()
			else:
				self.load(load_file)
		
		def create_grid_data(self):
			self.grid_size = [int((self.x_max - self.x_min) // self.min_grid_scale) + 1, int((self.y_max - self.y_min) / self.min_grid_scale) + 1]
			self.sorted_id = ti.field(dtype=ti.i32, shape=self.N * 10)
			self.grid_offset = ti.field(dtype=ti.i32, shape=self.grid_size)
			self.grid_offset_per_x = ti.field(dtype=ti.i32, shape=self.grid_size[0])
			self.grid_cnt = ti.field(dtype=ti.i32, shape=self.grid_size)
		
		@ti.kernel
		def reinitialize_grid_ti(self, positions: TiArr, grid_scale: ti.f32):
			for i in range(self.grid_size[0]):
				for j in range(self.grid_size[1]):
					self.grid_cnt[i, j] = 0
			for i in range(positions.shape[0]):
				if self.x_min <= positions[i, 0] <= self.x_max and self.y_min <= positions[i, 1] <= self.y_max:
					idx, idy = int((positions[i, 0] - self.x_min) // grid_scale), int((positions[i, 1] - self.y_min) // grid_scale)
					self.grid_cnt[idx, idy] += 1
			self.grid_offset_per_x[0] = 0
			for i in range(1, self.grid_size[0]):
				self.grid_offset_per_x[i] = 0
				for j in range(self.grid_size[1]):
					self.grid_offset_per_x[i] += self.grid_cnt[i-1, j]
			for _ in range(1):
				for i in range(1, self.grid_size[0]):
					self.grid_offset_per_x[i] += self.grid_offset_per_x[i-1]
			for i in range(self.grid_size[0]):
				self.grid_offset[i, 0] = self.grid_offset_per_x[i]
				for j in range(1, self.grid_size[1]):
					self.grid_offset[i, j] = self.grid_offset[i, j-1] + self.grid_cnt[i, j-1]
			for i in range(self.grid_size[0]):
				for j in range(self.grid_size[1]):
					self.grid_cnt[i, j] = 0
			for i in range(positions.shape[0]):
				if self.x_min <= positions[i, 0] <= self.x_max and self.y_min <= positions[i, 1] <= self.y_max:
					idx, idy = int((positions[i, 0] - self.x_min) // grid_scale), int((positions[i, 1] - self.y_min) // grid_scale)
					cur_cnt = ti.atomic_add(self.grid_cnt[idx, idy], 1)
					self.sorted_id[self.grid_offset[idx, idy] + cur_cnt] = i
		
		def reinitialize_grid(self):
			if self.clamp_threshold:
				self.grid_scale = max(np.sqrt(-2. * np.log(self.clamp_threshold)) * np.exp(-self.scalings.min().item()), self.min_grid_scale)
			else:
				self.grid_scale = max(self.x_max - self.x_min, self.y_max - self.y_min)
			self.reinitialize_grid_ti(self.positions.detach(), self.grid_scale)
		
		def parameters(self):
			return {
				'positions': self.positions,
				'scalings': self.scalings,
				'rotations': self.rotations,
				'values': self.values,
				'clamp_threshold': self.clamp_threshold,
				'min_grid_scale': self.min_grid_scale,
				'domain_range': (self.x_min, self.x_max, self.y_min, self.y_max)
			}
		
		def load(self, filename):
			parameters_dict = torch.load(filename)
			self.positions = parameters_dict['positions']
			self.scalings = parameters_dict['scalings']
			self.rotations = parameters_dict['rotations']
			self.values = parameters_dict['values']
			self.N = self.positions.shape[0]
			self.dim = self.values.shape[1]
			self.clamp_threshold = parameters_dict['clamp_threshold']
			self.min_grid_scale = parameters_dict['min_grid_scale']
			self.x_min, self.x_max, self.y_min, self.y_max = parameters_dict['domain_range']
			self.create_grid_data()
			self.zero_grad()
		
		@ti.kernel
		def get_losses_ti(self,
					   positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr,
					   grid_scale: ti.f32,
					   x: TiArr,
					   ref: TiArr, weight: ti.f32,
					   normals: TiArr, normal_ref: TiArr, weight_boundary: ti.f32,
					   val: TiArr,
					   stop_gradient: TiArr,
					   pde_parts: TiArr,
					   pde_parts_per_point: TiArr,  # 新增参数
					   dim: ti.i32): 
			#forward pass
			m = x.shape[0]
			# 初始化PDE loss的三个部分
			pde_parts[0] = 0.0  # A = Σ_i v_i ∂G_i(x)/∂t
			pde_parts[1] = 0.0  # B = Σ_i (G_i(x)-c)v_i
			pde_parts[2] = 0.0  # C = Σ_i v_i ∂G_i(x)/∂x
			for i in range(m):
				a = 0.0
				b = 0.0
				c = 0.0
				for d in range(dim):
					val[i, d] = 0.
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
							S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
							if gaussian >= self.clamp_threshold:    # within range
								for d in range(dim):
									val[i, d] += values[j_id, d] * (gaussian - self.clamp_threshold)
								# 计算PDE loss的三个部分
								if x.shape[1] >= 2:
									grad_gaussian = -gaussian * (cov_inv @ delta_pos)
									for d in range(dim):
										a += values[j_id, d] * grad_gaussian[1]
									for d in range(dim):
										b += values[j_id, d] * (gaussian - self.clamp_threshold)
									for d in range(dim):
										c += values[j_id, d] * grad_gaussian[0]
				# 存储每个点的A/B/C
				pde_parts_per_point[i, 0] = a
				pde_parts_per_point[i, 1] = b
				pde_parts_per_point[i, 2] = c
				# 同时累加到全局
				pde_parts[0] += a
				pde_parts[1] += b
				pde_parts[2] += c
			#backward pass
			if weight == 0 and weight_boundary == 0:    # disable backward
				m = 0
			# (warning: need atomic '+=')
			for i in range(m):
				is_boundary_point = (abs(x[i, 0] + 1.0) < 1e-5) or (abs(x[i, 0] - 1.0) < 1e-5)
				is_initial_point = (abs(x[i, 1]) < 1e-5)
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							if stop_gradient[j_id]:
								continue
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
							S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
							grad_gaussian = -gaussian * (cov_inv @ delta_pos)
							if gaussian >= self.clamp_threshold:	# within range
								# val_dot_normal = 0.
								# for d in range(dim):
								# 	val_dot_normal += val[i, d] * normals[i, d]
								# sign_val_dot_normal_diff = tm.sign(val_dot_normal - normal_ref[i])
								# maybe we dont need this part?

								# derivative w.r.t. values
								# derivative of loss
								values.grad[j_id, 0] += grad_gaussian[1]+(gaussian-self.clamp_threshold)*pde_parts_per_point[i, 2]+pde_parts_per_point[i, 1]*grad_gaussian[0]
								# derivative of loss_boundary_condition
								if is_boundary_point:
									values.grad[j_id, 0] += weight_boundary / m *2.0*val[i,0]*(gaussian-self.clamp_threshold)
								if is_initial_point:
									values.grad[j_id, 0] +=weight_boundary/m* 2*(val[i,0]+tm.sin(tm.pi*x[i,0]))*(gaussian-self.clamp_threshold)
								# derivative w.r.t. positions
								positions.grad[j_id, 0] += weight / (2. * m) * (values[j_id,0]*(grad_gaussian@cov_inv+gaussian@cov_inv@tm.vec2(1.,0.)@(cov_inv@delta_pos).transpose)+values[j_id,0]*gaussian@cov_inv@delta_pos*pde_parts_per_point[i, 2]+pde_parts_per_point[i,1]*values[j_id,0]*(grad_gaussian@cov_inv+gaussian@cov_inv@tm.vec2(0.,1.)@(cov_inv@delta_pos).transpose))[0]
								positions.grad[j_id, 1] += weight / (2. * m) * (values[j_id,1]*(grad_gaussian@cov_inv+gaussian@cov_inv@tm.vec2(0.,1.)@(cov_inv@delta_pos).transpose)+values[j_id,1]*gaussian@cov_inv@delta_pos*pde_parts_per_point[i, 2]+pde_parts_per_point[i,1]*values[j_id,1]*(grad_gaussian@cov_inv+gaussian@cov_inv@tm.vec2(0.,1.)@(cov_inv@delta_pos).transpose))[1]
								# derivative of loss_boundary
								if is_boundary_point:
									positions.grad[j_id, 0] +=2*val[i,0]*values[j_id,0]*gaussian*cov_inv@delta_pos[0]
									positions.grad[j_id, 1] +=2*val[i,0]*values[j_id,0]*gaussian*cov_inv@delta_pos[1]
								# derivative of initial_point
								if is_initial_point:
									positions.grad[j_id, 0] +=weight_boundary/m*2.0*(val[i,0]+tm.sin(tm.pi*x[i,0]))*values[j_id,0]*gaussian*cov_inv@delta_pos[0]
									positions.grad[j_id, 1] +=weight_boundary/m*2.0*(val[i,0]+tm.sin(tm.pi*x[i,0]))*values[j_id,0]*gaussian*cov_inv@delta_pos[1]
								# derivative w.r.t. scalings
								# derivative of loss
								M_x = tm.mat2([[2.0 * tm.exp(2.0 * scalings[j_id, 0]), 0.0], [0.0, 0.0]])
								d_cov_inv_ds_x = R @ M_x @ R.transpose()
								d_cov_inv_ds_delta_x = d_cov_inv_ds_x @ delta_pos
								delta_d_cov_inv_ds_delta_x = delta_pos.dot(d_cov_inv_ds_delta_x)
								a_i = cov_inv @ delta_pos 
								B_i = pde_parts_per_point[i, 1]
								C_i = pde_parts_per_point[i, 2]
								term1_x = -0.5 * delta_d_cov_inv_ds_delta_x * (a_i[1] + C_i)
								term2_x = -d_cov_inv_ds_delta_x[1]
								term3_x = B_i * (-0.5 * delta_d_cov_inv_ds_delta_x * a_i[0] - d_cov_inv_ds_delta_x[0])
								total_x = term1_x + term2_x + term3_x

								scalings.grad[j_id, 0] += weight / (2. * m) * values[j_id, 0] * gaussian * total_x

								M_y = tm.mat2([[0.0, 0.0], [0.0, 2.0 * tm.exp(2.0 * scalings[j_id, 1])]])
								d_cov_inv_ds_y = R @ M_y @ R.transpose()
								d_cov_inv_ds_delta_y = d_cov_inv_ds_y @ delta_pos
								delta_d_cov_inv_ds_delta_y = delta_pos.dot(d_cov_inv_ds_delta_y)

								term1_y = -0.5 * delta_d_cov_inv_ds_delta_y * (a_i[1] + C_i)
								term2_y = -d_cov_inv_ds_delta_y[1]
								term3_y = B_i * (-0.5 * delta_d_cov_inv_ds_delta_y * a_i[0] - d_cov_inv_ds_delta_y[0])
								total_y = term1_y + term2_y + term3_y

								scalings.grad[j_id, 1] += weight / (2. * m) * values[j_id, 0] * gaussian * total_y
								# derivative of loss_boundary
								if is_boundary_point:
									for k in range(2):  # k=0,1
										r_k = tm.vec2(R[0, k], R[1, k])
										dot_rk_delta = r_k.dot(delta_pos)
										exp_2s = tm.exp(2.0 * scalings[j_id, k])
										scalings.grad[j_id, k] += -2.0 * val[i,0] * values[j_id, 0] * gaussian * exp_2s * (dot_rk_delta ** 2)
								# derivative of initial_point
								if is_initial_point:
									for k in range(2):  # k=0,1
										r_k = tm.vec2(R[0, k], R[1, k])
										dot_rk_delta = r_k.dot(delta_pos)
										exp_2s = tm.exp(2.0 * scalings[j_id, k])
										scalings.grad[j_id, k] += -2.0 * (val[i,0]+tm.sin(tm.pi*x[i,0])) * values[j_id, 0] * gaussian * exp_2s * (dot_rk_delta ** 2)
								
								# derivative w.r.t. rotations
								# derivative of loss
								dR_dtheta = tm.mat2([
									[-tm.sin(rotations[j_id]), -tm.cos(rotations[j_id])],
									[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])]
								])
								d_cov_inv_dtheta = dR_dtheta @ S2 @ R.transpose() + R @ S2 @ dR_dtheta.transpose()
								# 计算中间量
								d_cov_inv_dtheta_delta = d_cov_inv_dtheta @ delta_pos
								alpha_i = delta_pos.dot(d_cov_inv_dtheta_delta)
								beta_i_x = d_cov_inv_dtheta_delta[0]  # 对应e_x^T
								beta_i_t = d_cov_inv_dtheta_delta[1]  # 对应e_t^T
								B_i = pde_parts_per_point[i, 1]
								C_i = pde_parts_per_point[i, 2]
								a_i = cov_inv @ delta_pos
								a_i_x = a_i[0]
								a_i_t = a_i[1]
								term1 = 0.5 * alpha_i * a_i_t - beta_i_t
								term2 = B_i * (0.5 * alpha_i * a_i_x - beta_i_x)
								term3 = -0.5 * alpha_i * C_i
								rot_grad = values[j_id, 0] * gaussian * (term1 + term2 + term3)
								rotations.grad[j_id] += weight / (2. * m) * rot_grad
								# derivative of loss_boundary
								if is_boundary_point:
									rotations.grad[j_id]+=weight_boundary/m *(-2.0*val[i,0]*values[j_id,0]*gaussian*delta_pos.transpose()@d_cov_inv_dtheta@delta_pos)
								# derivative of initial_point
								if is_initial_point:
									rotations.grad[j_id]+=weight_boundary/m *(-2.0*(val[i,0]+tm.sin(tm.pi*x[i,0]))*values[j_id,0]*gaussian*delta_pos.transpose()@d_cov_inv_dtheta@delta_pos)
		
		def get_losses(self, x, ref=None, weight=0., normals=None, normal_ref=None, weight_boundary=0., stop_gradient=None):
			if ref is None:
				ref = torch.zeros((x.shape[0], self.dim), device=device)
				weight = 0.
			if normals is None or normal_ref is None:
				normals = torch.zeros((x.shape[0], self.dim), device=device)
				normal_ref = torch.zeros(x.shape[0], device=device)
				weight_boundary = 0.
			if stop_gradient is None:
				stop_gradient = torch.zeros((self.positions.shape[0],), dtype=torch.int32, device=device)
			val = torch.zeros((x.shape[0], self.dim), device=device)
			
			# 创建PDE parts存储数组
			pde_parts = torch.zeros(3, device=device)  # [A, B, C]
			# 新增：每个点的ABC loss
			pde_parts_per_point = torch.zeros((x.shape[0], 3), device=device)
			
			self.get_losses_ti(
				self.positions, self.scalings, self.rotations, self.values,
				self.grid_scale,
				x,
				ref, weight,
				normals, normal_ref, weight_boundary,
				val,
				stop_gradient,
				pde_parts,
				pde_parts_per_point,  # 新增参数
				self.dim)
			
			# 将计算结果存储到类属性中
			self.pde_loss_parts['A'] = pde_parts[0].item()
			self.pde_loss_parts['B'] = pde_parts[1].item()
			self.pde_loss_parts['C'] = pde_parts[2].item()
			# 新增：存储每个点的ABC loss
			self.pde_loss_parts_per_point = pde_parts_per_point.cpu().numpy()
			return val, pde_parts_per_point
		
		def get_pde_loss_parts(self):
			"""
			获取PDE loss的三个部分
			"""
			return self.pde_loss_parts.copy()
		
		def print_pde_parts(self):
			"""
			打印PDE loss的三个部分
			"""
			print(f"A: {self.pde_loss_parts['A']:.6f}, B: {self.pde_loss_parts['B']:.6f}, C: {self.pde_loss_parts['C']:.6f}")
		
		def __call__(self, x):
			return self.get_losses(x)
		
		@ti.kernel
		def get_grad_losses_ti(self,
							positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr,
							grid_scale: ti.f32,
							x: TiArr,
							ref_grad: TiArr, weight_grad: ti.f32,
							ref_vor: TiArr, weight_vor: ti.f32,
							weight_div: ti.f32,
							grad: TiArr,
							vor_positions_grad: TiArr, vor_scalings_grad: TiArr, vor_rotations_grad: TiArr, vor_values_grad: TiArr,
							div_positions_grad: TiArr, div_scalings_grad: TiArr, div_rotations_grad: TiArr, div_values_grad: TiArr,
							stop_gradient: TiArr):
			m = x.shape[0]
			for i in range(m):
				for d in range(self.dim):
					grad[i, d, 0] = grad[i, d, 1] = 0.
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
							S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
							if gaussian >= self.clamp_threshold:	# within range
								grad_gaussian = -gaussian * (cov_inv @ delta_pos)
								for d in range(self.dim):
									grad[i, d, 0] += values[j_id, d] * grad_gaussian[0]
									grad[i, d, 1] += values[j_id, d] * grad_gaussian[1]
			if weight_grad == 0 and weight_vor == 0 and weight_div == 0:	# enable backward
				m = 0
			for i in range(m):
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							if stop_gradient[j_id]:
								continue
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
							S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
							if gaussian >= self.clamp_threshold:	# within range
								grad_gaussian = -gaussian * (cov_inv @ delta_pos)
								
								sign_vor_diff = 0.
								div2 = 0.
								value = tm.vec2(0., 0.)
								if self.dim == 2:
									sign_vor_diff = tm.sign((grad[i, 1, 0] - grad[i, 0, 1]) - ref_vor[i])
									div2 = 2. * (grad[i, 0, 0] + grad[i, 1, 1])
									value[0], value[1] = values[j_id, 0], values[j_id, 1]
								# derivative w.r.t. values
								for d in range(self.dim):
									# derivative of loss_grad
									values.grad[j_id, d] += weight_grad / (4. * m) * tm.sign(tm.vec2(grad[i, d, 0], grad[i, d, 1]) - tm.vec2(ref_grad[i, d, 0], ref_grad[i, d, 1])).dot(grad_gaussian)
								if self.dim == 2:
									# derivative of loss_vor
									vor_values_grad[j_id, 0] += weight_vor / m * sign_vor_diff * -grad_gaussian[1]
									vor_values_grad[j_id, 1] += weight_vor / m * sign_vor_diff * grad_gaussian[0]
									# derivative of loss_div
									div_values_grad[j_id, 0] += weight_div / m * div2 * grad_gaussian[0]
									div_values_grad[j_id, 1] += weight_div / m * div2 * grad_gaussian[1]
								
								# derivative w.r.t. positions
								d_grad_gaussian_position = gaussian * cov_inv @ (tm.eye(2) - delta_pos.outer_product(delta_pos) @ cov_inv)
								sign_times_value = tm.vec2(0., 0.)
								for d in range(self.dim):
									sign_times_value += tm.sign(tm.vec2(grad[i, d, 0] - ref_grad[i, d, 0], grad[i, d, 1] - ref_grad[i, d, 1])) * values[j_id, d]
								sign_dvor_times_value = sign_vor_diff * tm.mat2([[0., 1.], [-1., 0.]]) @ value
								sign_div2_times_value = div2 * value
								# derivative of loss_grad
								positions.grad[j_id, 0] += weight_grad / (4. * m) * d_grad_gaussian_position[:, 0].dot(sign_times_value)
								positions.grad[j_id, 1] += weight_grad / (4. * m) * d_grad_gaussian_position[:, 1].dot(sign_times_value)
								if self.dim == 2:
									# derivative of loss_vor
									vor_positions_grad[j_id, 0] += weight_vor / m * d_grad_gaussian_position[:, 0].dot(sign_dvor_times_value)
									vor_positions_grad[j_id, 1] += weight_vor / m * d_grad_gaussian_position[:, 1].dot(sign_dvor_times_value)
									# derivative of loss_div
									div_positions_grad[j_id, 0] += weight_div / m * d_grad_gaussian_position[:, 0].dot(sign_div2_times_value)
									div_positions_grad[j_id, 1] += weight_div / m * d_grad_gaussian_position[:, 1].dot(sign_div2_times_value)
								
								# derivative w.r.t. scalings
								vec_cos_sin = tm.vec2(tm.cos(rotations[j_id]), tm.sin(rotations[j_id]))
								d_grad_gaussian_scaling_0 = -(-gaussian * tm.exp(2. * scalings[j_id, 0]) * (vec_cos_sin.dot(delta_pos)) ** 2) * cov_inv @ delta_pos - gaussian * (2. * tm.exp(2. * scalings[j_id, 0]) * vec_cos_sin.outer_product(vec_cos_sin)) @ delta_pos
								vec_negsin_cos = tm.vec2(-tm.sin(rotations[j_id]), tm.cos(rotations[j_id]))
								d_grad_gaussian_scaling_1 = -(-gaussian * tm.exp(2. * scalings[j_id, 1]) * (vec_negsin_cos.dot(delta_pos)) ** 2) * cov_inv @ delta_pos - gaussian * (2. * tm.exp(2. * scalings[j_id, 1]) * vec_negsin_cos.outer_product(vec_negsin_cos)) @ delta_pos
								# derivative of loss_grad
								scalings.grad[j_id, 0] += weight_grad / (4. * m) * d_grad_gaussian_scaling_0.dot(sign_times_value)
								scalings.grad[j_id, 1] += weight_grad / (4. * m) * d_grad_gaussian_scaling_1.dot(sign_times_value)
								if self.dim == 2:
									# derivative of loss_vor
									vor_scalings_grad[j_id, 0] += weight_vor / m * d_grad_gaussian_scaling_0.dot(sign_dvor_times_value)
									vor_scalings_grad[j_id, 1] += weight_vor / m * d_grad_gaussian_scaling_1.dot(sign_dvor_times_value)
									# derivative of loss_div
									div_scalings_grad[j_id, 0] += weight_div / m * d_grad_gaussian_scaling_0.dot(sign_div2_times_value)
									div_scalings_grad[j_id, 1] += weight_div / m * d_grad_gaussian_scaling_1.dot(sign_div2_times_value)
								
								# derivative w.r.t. rotations
								d_cov_inv_rotation = (tm.exp(2. * scalings[j_id, 0]) - tm.exp(2. * scalings[j_id, 1])) * tm.mat2([[-tm.sin(2. * rotations[j_id]), tm.cos(2. * rotations[j_id])], [tm.cos(2. * rotations[j_id]), tm.sin(2. * rotations[j_id])]])
								d_grad_gaussian_rotation = -(-.5 * gaussian * (delta_pos.outer_product(delta_pos) @ d_cov_inv_rotation).trace()) * cov_inv @ delta_pos - gaussian * d_cov_inv_rotation @ delta_pos
								# derivative of loss_grad
								rotations.grad[j_id] += weight_grad / (4. * m) * d_grad_gaussian_rotation.dot(sign_times_value)
								if self.dim == 2:
									# derivative of loss_vor
									vor_rotations_grad[j_id] += weight_vor / m * d_grad_gaussian_rotation.dot(sign_dvor_times_value)
									# derivative of loss_div
									div_rotations_grad[j_id] += weight_div / m * d_grad_gaussian_rotation.dot(sign_div2_times_value)
		
		def get_grad_losses(self, x,
							ref_grad=None, weight_grad=0.,
							ref_vor=None, weight_vor=0.,
							weight_div=0.,
							vor_positions_grad=None, vor_scalings_grad=None, vor_rotations_grad=None, vor_values_grad=None,
							div_positions_grad=None, div_scalings_grad=None, div_rotations_grad=None, div_values_grad=None,
							stop_gradient=None):
			if ref_grad is None:
				ref_grad = torch.zeros((x.shape[0], self.dim, 2), device=device)
				weight_grad = 0.
			if ref_vor is None:
				ref_vor = torch.zeros((x.shape[0],), device=device)
				weight_vor = 0.
			if stop_gradient is None:
				stop_gradient = torch.zeros((self.positions.shape[0],), dtype=torch.int32, device=device)
			grad = torch.zeros((x.shape[0], self.dim, 2), device=device)
			if vor_positions_grad is None:
				vor_positions_grad = self.positions.grad
			if vor_scalings_grad is None:
				vor_scalings_grad = self.scalings.grad
			if vor_rotations_grad is None:
				vor_rotations_grad = self.rotations.grad
			if vor_values_grad is None:
				vor_values_grad = self.values.grad
			if div_positions_grad is None:
				div_positions_grad = self.positions.grad
			if div_scalings_grad is None:
				div_scalings_grad = self.scalings.grad
			if div_rotations_grad is None:
				div_rotations_grad = self.rotations.grad
			if div_values_grad is None:
				div_values_grad = self.values.grad
			self.get_grad_losses_ti(
				self.positions, self.scalings, self.rotations, self.values,
				self.grid_scale,
				x,
				ref_grad, weight_grad,
				ref_vor, weight_vor,
				weight_div,
				grad,
				vor_positions_grad, vor_scalings_grad, vor_rotations_grad, vor_values_grad,
				div_positions_grad, div_scalings_grad, div_rotations_grad, div_values_grad,
				stop_gradient)
			return grad
		
		def gradient(self, x, need_val=False):
			grad = self.get_grad_losses(x)
			return (grad, self.__call__(x)) if need_val else grad
		
		@ti.func
		def get_2d_val_grad_ti(self, positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr, grid_scale: ti.f32, x: tm.vec2):
			val = tm.vec2(0.)
			grad = tm.mat2(0.)
			idx, idy = int((x[0] - self.x_min) // grid_scale), int((x[1] - self.y_min) // grid_scale)
			for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
				for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
					for j in range(ti.i32(self.grid_cnt[gi, gj])):
						j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
						delta_pos = tm.vec2(x[0] - positions[j_id, 0], x[1] - positions[j_id, 1])
						R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
						S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
						cov_inv = R @ S2 @ R.transpose()
						gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
						if gaussian >= self.clamp_threshold:	# within range
							grad_gaussian = -gaussian * (cov_inv @ delta_pos)
							for d in range(self.dim):
								val[d] += values[j_id, d] * (gaussian - self.clamp_threshold)
								grad[d, 0] += values[j_id, d] * grad_gaussian[0]
								grad[d, 1] += values[j_id, d] * grad_gaussian[1]
			return val, grad
		
		@ti.kernel
		def advection_rk4_ti(self,
							positions: TiArr, scalings: TiArr, rotations: TiArr, values: TiArr, grid_scale: ti.f32,
							start_pos: TiArr, dt: ti.f32,
							goal_pos: TiArr, deformation: TiArr, goal_val: TiArr, goal_grad: TiArr):
			for i in range(start_pos.shape[0]):
				x = tm.vec2(start_pos[i, 0], start_pos[i, 1])
				v, dv = self.get_2d_val_grad_ti(positions, scalings, rotations, values, grid_scale, x)
				phi1 = x + dt * .5 * v
				v1, dv1 = self.get_2d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi1)
				phi2 = x + dt * .5 * v1
				v2, dv2 = self.get_2d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi2)
				phi3 = x + dt * v2
				v3, dv3 = self.get_2d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi3)
				phi = x + dt / 6. * (v + 2. * v1 + 2. * v2 + v3)
				goal_pos[i, 0], goal_pos[i, 1] = phi[0], phi[1]
				if deformation.shape[0] > 0:
					dphi1 = tm.eye(2) + dt * .5 * dv
					dv1_x_dphi1 = dv1 @ dphi1
					dphi2 = tm.eye(2) + dt * .5 * dv1_x_dphi1
					dv2_x_dphi2 = dv2 @ dphi2
					dphi3 = tm.eye(2) + dt * dv2_x_dphi2
					dphi = tm.eye(2) + dt / 6. * (dv + 2. * dv1_x_dphi1 + 2. * dv2_x_dphi2 + dv3 @ dphi3)
					for j in range(2):
						for k in range(2):
							deformation[i, j, k] = dphi[j, k]
				if goal_val.shape[0] > 0 and goal_grad.shape[0] > 0:
					v_phi, dv_phi = self.get_2d_val_grad_ti(positions, scalings, rotations, values, grid_scale, phi)
					goal_val[i, 0], goal_val[i, 1] = v_phi[0], v_phi[1]
					for j in range(2):
						for k in range(2):
							goal_grad[i, j, k] = dv_phi[j, k]
		
		def advection_rk4(self, start_pos, dt, pos_only=True):
			goal_pos = torch.zeros_like(start_pos, device=device)
			deformation = torch.zeros((0 if pos_only else start_pos.shape[0], 2, 2), device=device)
			goal_val = torch.zeros((0 if pos_only else start_pos.shape[0], 2), device=device)
			goal_grad = torch.zeros((0 if pos_only else start_pos.shape[0], 2, 2), device=device)
			self.advection_rk4_ti(
				self.positions, self.scalings, self.rotations, self.values, self.grid_scale,
				start_pos, dt,
				goal_pos, deformation, goal_val, goal_grad
			)
			return goal_pos if pos_only else (goal_pos, deformation, goal_val, goal_grad)
		
		@ti.kernel
		def get_coverage_ti(self,
					   positions: TiArr, scalings: TiArr, rotations: TiArr,
					   grid_scale: ti.f32,
					   x: TiArr, res: TiArr):
			m = x.shape[0]
			for i in range(m):
				res[i] = 0.
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							R = tm.mat2([[tm.cos(rotations[j_id]), -tm.sin(rotations[j_id])], [tm.sin(rotations[j_id]), tm.cos(rotations[j_id])]])
							S2 = tm.mat2([[tm.exp(2. * scalings[j_id, 0]), 0.], [0., tm.exp(2. * scalings[j_id, 1])]])
							cov_inv = R @ S2 @ R.transpose()
							gaussian = tm.exp(-.5 * delta_pos @ cov_inv @ delta_pos)
							if gaussian >= self.clamp_threshold:	# within range
								res[i] += gaussian - self.clamp_threshold
		
		def get_coverage(self, x):
			res = torch.zeros(x.shape[0], device=device)
			self.get_coverage_ti(self.positions, self.scalings, self.rotations, self.grid_scale, x, res)
			return res
		
		@ti.kernel
		def get_all_neighbors_ti(self, x: TiArr, positions: TiArr, grid_scale: ti.f32, mark: TiArr):
			for i in range(x.shape[0]):
				idx, idy = int((x[i, 0] - self.x_min) // grid_scale), int((x[i, 1] - self.y_min) // grid_scale)
				for gi in range(max(idx - 1, 0), min(idx + 1, self.grid_size[0] - 1) + 1):
					for gj in range(max(idy - 1, 0), min(idy + 1, self.grid_size[1] - 1) + 1):
						for j in range(ti.i32(self.grid_cnt[gi, gj])):
							j_id = self.sorted_id[self.grid_offset[gi, gj] + j]
							delta_pos = tm.vec2(x[i, 0] - positions[j_id, 0], x[i, 1] - positions[j_id, 1])
							if tm.length(delta_pos) <= grid_scale:
								mark[j_id] = 1
		
		def get_all_neighbors(self, x):
			mark = torch.zeros((self.positions.shape[0],), dtype=torch.int32, device=device)
			self.get_all_neighbors_ti(x, self.positions, self.grid_scale, mark)
			return mark.bool()
		
		def zero_grad(self):
			for param in super().parameters().values():
				if param.grad is None:
					param.grad = torch.zeros_like(param, device=device)
				else:
					param.grad.zero_()
			self.reinitialize_grid()
		
		def step(self, metrics):
			super().step(metrics)
			self.zero_grad()


def generate_blue_noise(n, x_min=0., x_max=1., y_min=0., y_max=1.):
	x_scale, y_scale = x_max - x_min, y_max - y_min
	samples = np.zeros((n, 2))
	samples[0] = np.random.random(2) * np.array([x_scale, y_scale]) + np.array([x_min, y_min])
	for i in range(1, n):
		best_candidate = None
		best_distance = 0
		for _ in range(50):
			candidate = np.random.random(2) * np.array([x_scale, y_scale]) + np.array([x_min, y_min])
			distance = ((candidate - samples[:i, :]) ** 2).sum(axis=1).min()
			if distance > best_distance:
				best_candidate = candidate
				best_distance = distance
		samples[i] = best_candidate
	return samples


def get_grid_points(x_min, x_max, y_min, y_max, x_N, y_N):
	X = torch.linspace(x_min, x_max, x_N, device=device)
	Y = torch.linspace(y_min, y_max, y_N, device=device)
	X_, Y_ = torch.meshgrid(X, Y, indexing='xy')
	XY = torch.stack((X_, Y_)).permute(1, 2, 0).reshape(-1, 2)
	return XY.contiguous()


def show_field(field, x_min, x_max, y_min, y_max, dim=1, x_N=100, y_N=100, additional_drawing=None, plt_show=True, save_filename=None):
	if dim == 1:
		XY = get_grid_points(x_min, x_max, y_min, y_max, x_N, y_N)
		H = field(XY).reshape(y_N, x_N)
		plt.axis('equal')
		plt.imshow(H.detach().cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='jet')
		plt.colorbar()
	else:
		XY_ = get_grid_points(x_min, x_max, y_min, y_max, x_N, y_N)
		UV = field(XY_) # XY_ may change after calling field()
		X_, Y_ = XY_[:, 0], XY_[:, 1]
		U, V = UV[:, 0], UV[:, 1]
		non_zero = (U**2 + V**2) != 0
		plt.axis('equal')
		if non_zero.any():
			plt.quiver(X_.cpu(), Y_.cpu(), U.detach().cpu(), V.detach().cpu())
	if additional_drawing:
		additional_drawing()
	if save_filename is None:
		if plt_show:
			plt.show()
	else:
		plt.savefig(save_filename)
		plt.clf()


def draw_ellipses(gaussian_splatting: GaussianSplatting, indices=None, scattering=True):
	if scattering:
		with torch.no_grad():
			plt.scatter(gaussian_splatting.positions[:, 0].cpu(), gaussian_splatting.positions[:, 1].cpu(), s=.5, color='red')
	ax = plt.gca()
	with torch.no_grad():
		for i in random.sample(list(range(gaussian_splatting.N)), min(20, gaussian_splatting.N)) if indices is None else indices:
			width, height = (1. / torch.exp(gaussian_splatting.scalings[i])).cpu()
			ellipse = Ellipse(gaussian_splatting.positions[i].cpu(), width, height, angle=gaussian_splatting.rotations[i].cpu() / torch.pi * 180., fill=False)
			ax.add_patch(ellipse)