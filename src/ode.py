import torch
import numpy as np

import math
from tqdm import tqdm
import typing as tp

from scipy import integrate, linalg
import sys
sys.path.append("/home/windowskonon1337/sources/FieldMatching")
from src.utils import from_flattened_numpy,  to_flattened_numpy


#############################
class DippoleGroundTrurthEFMODESolver:
    
    def __init__(self, config) -> None:
        self._config = config # private attribute
        
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self,config) -> None:
        print("You modify configuration for running code")
        self._config = config
    
    def __call__(self, efm ,
                       perturbed_samples_vec: torch.Tensor ,
                       p_samples: torch.Tensor,
                       q_samples: torch.Tensor) -> tp.Sequence[torch.Tensor]: ## ???
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()] 
        #### uniform motion along axis z ####
        for step in tqdm(range(math.ceil(self._config.L//self._config.ode.step))):  
            
            field =  efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec,
                                     p_samples=p_samples,
                                     q_samples=q_samples)
            
            perturbed_samples_vec = perturbed_samples_vec +\
                                    (self._config.ode.step/field[:,0].view(-1,1) + self._config.ode.gamma)*field 
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
        #### uniform motion along axis z ####
        
         
        #### movement behind the second plate ####
        field = efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec,
                                     p_samples=p_samples,
                                     q_samples=q_samples)
        mask_start = field[:,0] > 0 # E_z > 0
        mask = mask_start 
        
        #while  torch.nonzero(mask).__len__() != 0:
        for _ in tqdm(range(self._config.ode.behind_num_steps)):
            if not mask.any():
                break

            field[mask] = efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec[mask],
                                     p_samples=p_samples,
                                     q_samples=q_samples)
            perturbed_samples_vec[mask] = perturbed_samples_vec[mask] +\
                                          (self._config.ode.behind_step/torch.norm(field[mask],keepdim=True))*field[mask]
            
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = torch.logical_and(mask_start, (perturbed_samples_vec[:,0] >= self._config.q.x_loc + 0.05).view(-1))  
        #### movement behind the second plate ####
        return perturbed_samples_vec, trajectory
#############################






#############################
class LearnDippoleEFMODESolver:
    
    def __init__(self, net, config):
        
        self._config = config
        self.net = net
     
    @property
    def config(self):
        return slef._config
    
    @config.setter
    def config(self,config):
        self._config = config
    
    def __call__(self, perturbed_samples_vec: torch.Tensor ,
                        p_samples: torch.Tensor,
                        q_samples: torch.Tensor) -> tp.Sequence[torch.Tensor]:
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()] 
        #### uniform motion along axis z ####
        for step in tqdm(range(math.ceil(self._config.L//self._config.ode.step))):  
            
            field =  self.net(perturbed_samples_vec)
            print(type(self.net))
            perturbed_samples_vec = perturbed_samples_vec +\
                                    (self._config.ode.step/field[:,0].view(-1,1) + self._config.ode.gamma)*field 
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
        #### uniform motion along axis z ####
        
         
        #### movement behind the second plate ####
        field =  self.net(perturbed_samples_vec)
        mask_start = field[:,0] > 0 # E_z > 0
        mask = mask_start 
        
        #while  torch.nonzero(mask).__len__() != 0:
        for _ in tqdm(range(self._config.ode.behind_num_steps)):
            
             
            field[mask] = self.net(perturbed_samples_vec[mask])
            perturbed_samples_vec[mask] = perturbed_samples_vec[mask] +\
                                          (self._config.ode.behind_step/torch.norm(field[mask],keepdim=True))*field[mask]
            
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = torch.logical_and(mask_start, (perturbed_samples_vec[:,0] >=  self._config.q.x_loc + 0.05).view(-1))
             
        #### movement behind the second plate ####
        
        
        return perturbed_samples_vec, trajectory 
#############################







#############################
def get_rk45_sampler_pfgm( y, config, shape,   rtol=1e-4, atol=1e-4,
                    method='RK45', eps=1e-3, device='cuda'):

    """RK45 ODE sampler for PFGM.

    Args:
    sde: An `methods.SDE` object that represents PFGM.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

    Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
    """

    
    def ode_sampler(model, y):

        x = y

        z = torch.ones((len(x), 1, 1, 1)).to(x.device)
        z = z.repeat((1, 1, config.data.image_size, config.data.image_size)) * config.L
        x = x.view(shape)
        # Augment the samples with extra dimension z
        # We concatenate the extra dimension z as an addition channel to accomondate this solver
        x = torch.cat((z, x), dim=1)
        x = x.float()
        new_shape = (len(x), config.data.num_channels + 1, config.data.image_size, config.data.image_size)
        
       
         

        

        def ode_func(t, x):

           


            x = from_flattened_numpy(x, new_shape).to(device).type(torch.float32)

            # Change-of-variable z=exp(t)
            z = np.exp(t)
            #net_fn = get_predict_fn(sde, model, train=False)

            x_drift, z_drift = model(x[:, 1:], torch.ones((len(x))).to(device) * z)
            x_drift = x_drift.view(len(x_drift), -1)

            # Substitute the predicted z with the ground-truth
            # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
            z_exp = config.sampling.z_exp

 

            if z < z_exp and config.training.gamma > 0:
                data_dim = config.data.image_size * config.data.image_size * config.data.num_channels
                sqrt_dim = np.sqrt(data_dim)
                norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
                x_norm = config.training.gamma * norm_1 / (1 - norm_1)
                x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
                z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + config.training.gamma)

                
                
            # Predicted normalized Poisson field
            v = torch.cat([ z_drift[:, None], x_drift], dim=1)
            dt_dz = 1 / (v[:, 0] + 1e-5)
            dx_dt = v[:, 1:].view(shape)

            # Get dx/dz
            dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
            # drift = z * (dx/dz, dz/dz) = z * (dx/dz, 1)
            drift = torch.cat([torch.ones((len(dx_dz), 1, config.data.image_size,
                                           config.data.image_size)).to(dx_dz.device) * z, z * dx_dz], dim=1)
            return to_flattened_numpy(drift)

 
        
        

        
        # Black-box ODE solver for the probability flow ODE.
        # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
        solution = integrate.solve_ivp(ode_func,
                                       (np.log(config.L),
                                                  np.log(config.training.epsilon)), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)

        nfe = solution.nfev
        num_itrs = len(solution.y[0])
        x = torch.tensor(solution.y[:, -1]).reshape(new_shape).to(device).type(torch.float32)
        
        trajectory = []
        visual_iters = np.linspace(int(num_itrs//8), num_itrs,  10)
       
        for itr in visual_iters:
            traj = torch.tensor(solution.y[:,int(itr)-1]).reshape(new_shape).to(device).type(torch.float32)
            trajectory.append(traj[:,1:].detach().cpu())
            
            
        # Detach augmented z dimension
        x = x[:, 1:]
        #x = inverse_scaler(x)
        return x, nfe, torch.stack(trajectory,dim=0)
 
    return ode_sampler


############################

  
    

    
#############################
class LearnedImageODESolver:

    def __init__(self, net, config):
        self.config = config
        self.net = net

    def __call__(self, x_init ):
        trajectory = [x_init[:,1:].view(-1,self.config.data.num_channels,
                                               self.config.data.image_size,
                                               self.config.data.image_size).clone().detach().cpu()]
        mask = torch.tensor(x_init.shape[0]*[True]).to(self.config.device)
        
        while mask.any():
 
                
            field_x, field_z = self.net(x_init[:,1:].view(-1,self.config.data.num_channels,
                                                             self.config.data.image_size,
                                                             self.config.data.image_size) , x_init[:,0]   )
            
            field = torch.cat([field_z.view(-1,1),
                               field_x.view(-1, self.config.data.num_channels*\
                                                self.config.data.image_size*\
                                                self.config.data.image_size)], dim=1) # [B, 1+C*H*W]
            
            # backward
            x_init  = x_init  - (self.config.ode.step/ ( field_z.view(-1,1)  + self.config.ode.gamma ))*field # [B, C*H*W+1]
            trajectory.append(x_init[:,1:].view(-1,self.config.data.num_channels,
                                               self.config.data.image_size,
                                               self.config.data.image_size).clone().detach().cpu())
            t = x_init[:,0]
            mask = t[0] > self.config.training.epsilon
            #mask = t[0] < self.config.L - self.config.training.epsilon
          
            
            
            
        return x_init, trajectory
#############################   
    
    
    
    
    
    
    



#############################

class BaseODESolver:

    def __init__(self, config):
        self.config = config

    def __call__(self, func, perturbed_samples_vec , p_samples, q_samples):
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()]
        mask = torch.tensor(perturbed_samples_vec.shape[0]*[True]).to(self.config.device)
        
        while mask.any():
            field = func( perturbed_samples_vec,
                          p_samples, q_samples, self.config)
           
            perturbed_samples_vec  = perturbed_samples_vec  +\
            (self.config.ode.step/ (field[:,0][:,None] + self.config.ode.gamma ))*field

            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = perturbed_samples_vec[:,0] < self.config.L 
            #print(torch.min(perturbed_samples_vec[:,0]))
            
        return perturbed_samples_vec, trajectory
#############################











#############################
class LearnedODESolver:

    def __init__(self, net, config):
        self.config = config
        self.net = net

    def __call__(self, perturbed_samples_vec, p_samples, q_samples):
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()]
        mask = torch.tensor(perturbed_samples_vec.shape[0]*[True])
        
        while mask.any():
 
                
            field = self.net(perturbed_samples_vec)
            perturbed_samples_vec  = perturbed_samples_vec  +\
                                     (self.config.ode.step/ (field[:,0][:,None] + self.config.ode.gamma ))*field
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = perturbed_samples_vec[:,0] < self.config.L 

        return perturbed_samples_vec, trajectory
#############################


#############################
### WIP
class DynamicODESolver:
    """
    ODE solver для динамического уравнения движения частиц: mx" = qE
    
    Решает систему уравнений:
    - dx/dt = v (скорость)
    - dv/dt = (q/m)E(x) (ускорение)
    
    где:
    - m - масса частицы
    - q - заряд частицы
    - E(x) - электрическое поле в точке x
    """
    
    def __init__(self, config, mass=1.0, charge=1.0, initial_velocity=None):
        """
        Args:
            config: конфигурация с параметрами ODE
            mass: масса частицы (по умолчанию 1.0)
            charge: заряд частицы (по умолчанию 1.0)
            initial_velocity: начальная скорость [batch_size, DIM] или None (нулевая скорость)
        """
        self._config = config
        self.mass = mass
        self.charge = charge
        self.initial_velocity = initial_velocity
        
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config):
        print("You modify configuration for running code")
        self._config = config
    
    def __call__(self, efm,
                 perturbed_samples_vec: torch.Tensor,
                 p_samples: torch.Tensor,
                 q_samples: torch.Tensor) -> tp.Sequence[torch.Tensor]:
        batch_size = perturbed_samples_vec.shape[0]
        dim = perturbed_samples_vec.shape[1]
        device = perturbed_samples_vec.device
        
        if self.initial_velocity is None:
            velocity = torch.zeros_like(perturbed_samples_vec)
        else:
            velocity = self.initial_velocity.to(device)
            if velocity.shape != perturbed_samples_vec.shape:
                raise ValueError(f"Initial velocity shape {velocity.shape} doesn't match positions shape {perturbed_samples_vec.shape}")
        
        position = perturbed_samples_vec.clone()
        
        trajectory = [position.clone().detach().cpu()]
        
        acceleration_coeff = self.charge / self.mass
        
        dt = getattr(self._config.ode, 'dt', self._config.ode.step)
        
        num_steps = math.ceil(self._config.L // self._config.ode.step)
        
        for step in tqdm(range(num_steps)):
            field = efm.GroundTruth(
                perturbed_samples_vec=position,
                p_samples=p_samples,
                q_samples=q_samples
            )
            
            acceleration = acceleration_coeff * field
            
            velocity = velocity + acceleration * dt
            position = position + velocity * dt
            
            trajectory.append(position.clone().detach().cpu())
        
        behind_dt = getattr(self._config.ode, 'behind_dt', self._config.ode.behind_step)
        
        for _ in tqdm(range(self._config.ode.behind_num_steps)):
            field = efm.GroundTruth(
                perturbed_samples_vec=position,
                p_samples=p_samples,
                q_samples=q_samples
            )
            
            acceleration = acceleration_coeff * field
            
            velocity = velocity + acceleration * behind_dt
            position = position + velocity * behind_dt
            
            trajectory.append(position.clone().detach().cpu())
        
        return position, trajectory
#############################


#############################
class LatentODESolver:
    def __init__(self, field_network, config):
        self.field_network = field_network
        self.config = config
        self.device = config.device
    
    def sample(self, z_init, num_steps=None, method='euler'):
        if num_steps is None:
            num_steps = getattr(self.config.ode, 'num_steps', 100)
        
        batch_size = z_init.shape[0]
        latent_dim = z_init.shape[1]
        
        z_coord = self.config.L * torch.ones(batch_size).to(self.device)
        z_latent = z_init.to(self.device)
        
        dt = -self.config.L / num_steps 
        
        trajectory = []
        
        with torch.no_grad():
            for step in range(num_steps):
                trajectory.append(z_latent.clone().cpu())
                
                if method == 'euler':
                    field_z_coord, field_z_latent = self.field_network(z_coord, z_latent)
                    
                    z_coord = z_coord + field_z_coord * dt
                    z_latent = z_latent + field_z_latent * dt
                
                elif method == 'rk4':
                    k1_coord, k1_latent = self.field_network(z_coord, z_latent)
                    
                    z_coord_mid1 = z_coord + k1_coord * dt / 2
                    z_latent_mid1 = z_latent + k1_latent * dt / 2
                    k2_coord, k2_latent = self.field_network(z_coord_mid1, z_latent_mid1)
                    
                    z_coord_mid2 = z_coord + k2_coord * dt / 2
                    z_latent_mid2 = z_latent + k2_latent * dt / 2
                    k3_coord, k3_latent = self.field_network(z_coord_mid2, z_latent_mid2)
                    
                    z_coord_end = z_coord + k3_coord * dt
                    z_latent_end = z_latent + k3_latent * dt
                    k4_coord, k4_latent = self.field_network(z_coord_end, z_latent_end)
                    
                    z_coord = z_coord + (k1_coord + 2*k2_coord + 2*k3_coord + k4_coord) * dt / 6
                    z_latent = z_latent + (k1_latent + 2*k2_latent + 2*k3_latent + k4_latent) * dt / 6
                
                z_coord = torch.clamp(z_coord, 
                                      self.config.training.epsilon, 
                                      self.config.L - self.config.training.epsilon)
        
        trajectory.append(z_latent.clone().cpu())
        
        return z_latent, trajectory
    
    def sample_conditional(self, z_init, anchor_z, k_nearest=5):
        # TODO: implement cluster-conditioned field evaluation
        return self.sample(z_init)
#############################


