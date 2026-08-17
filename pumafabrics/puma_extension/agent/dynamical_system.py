from pumafabrics.puma_extension.agent.utils.dynamical_system_operations import batch_dot_product, denormalize_derivative, euler_integration, normalize_state, denormalize_state, get_derivative_normalized_state
import torch
import numpy as np


class DynamicalSystem():
    """
    Dynamical System that uses Neural Network trained with Contrastive Imitation
    """
    def __init__(self, x_init, space, order, min_state_derivative, max_state_derivative, saturate_transition, primitive_type,
                 model, dim_state, delta_t, x_min, x_max, radius):
        # Initialize NN model
        self.model = model

        # Initialize parameters
        self.space = space
        self.order = order
        self.saturate_transition = saturate_transition
        self.primitive_type = primitive_type
        self.dim_state = dim_state
        self.dim_position = dim_state // order
        self.min_vel = min_state_derivative[0]
        self.max_vel = max_state_derivative[0]
        self.max_vel_norm = torch.max(-self.min_vel, self.max_vel)  # axes are treated independently
        self.min_acc = min_state_derivative[1]
        self.max_acc = max_state_derivative[1]
        if self.min_acc is not None:
            self.max_acc_norm = torch.max(-self.min_acc, self.max_acc)  # axes are treated independently
        self.delta_t = delta_t
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.radius = radius
        self.batch_size = x_init.shape[0]

        # Project points to sphere surface
        if self.space == 'sphere':
            x_init = self.map_points_to_sphere(x_init)
        elif self.space == 'euclidean_sphere':
            projected_points = self.map_points_to_sphere(x_init[:, 3:self.dim_position])
            x_init = torch.cat([x_init[:, :3], projected_points, x_init[:, self.dim_position:]], dim=1)  # pytorch doesn't like inplace operations

        # Init dynamical system state
        self.x_t_d = x_init
        self.y_t = self.model.encoder(x_init, self.primitive_type)

    def map_points_to_sphere(self, x_t, radius=1):
        """
        Projects points to sphere surface
        """
        norm = torch.linalg.norm(x_t, dim=1).reshape(-1, 1)
        x_t = (radius / norm) * x_t
        return x_t

    def map_to_derivative(self, y_t):
        """
        Maps latent state to task state derivative
        """
        # Get desired velocity (phi)
        dx_t_d_normalized = self.model.decoder_dx(y_t)

        # Denormalize velocity/acceleration
        if self.order == 1:
            dx_t_d = denormalize_derivative(dx_t_d_normalized, self.max_vel_norm)
        elif self.order == 2:
            dx_t_d = denormalize_derivative(dx_t_d_normalized, self.max_acc_norm)
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        return dx_t_d

    def project_point_onto_plane(self, p, n, r=0):
        """
        Projects a point p onto a plane defined by point r and normal vector n in R^{n+1}.

        Args:
        p (torch.Tensor): The point to project, shape (n+1,)
        r (torch.Tensor): A point on the plane, shape (n+1,)
        n (torch.Tensor): The normal vector of the plane, shape (n+1,)

        Returns:
        torch.Tensor: The projection of p onto the plane.
        """

        # Compute the vector from r to p
        v = p - r

        # Calculate the dot product of v and n
        dot_v_n = batch_dot_product(v, n)

        # Calculate the dot product of n with itself
        dot_n_n = batch_dot_product(n, n)

        # Calculate the projection of v onto n
        proj_v_onto_n = (dot_v_n / dot_n_n) * n

        # Calculate the projection of p onto the plane
        p_plane = p - proj_v_onto_n

        return p_plane

    def exp_map_sphere(self, p, v):
        v_norm = v.norm(dim=1, keepdim=True)
        epsilon = 1e-8
        safe_v_norm = v_norm.clamp(min=epsilon)
        mapped_point = torch.cos(v_norm) * p + torch.sin(v_norm) * (v / safe_v_norm)
        return mapped_point

    def integrate_non_euclidean_1st_order(self, x_t, vel_t_d):
        # Project velocity to tangent space
        vel_t_d = self.project_point_onto_plane(vel_t_d, x_t)

        # Compute exponential map
        delta_x_d_tangent = vel_t_d * self.delta_t
        x_t_d = self.exp_map_sphere(x_t, delta_x_d_tangent)
        return x_t_d, vel_t_d

    def integrate_non_euclidean_2nd_order(self, x_t, vel_t, acc_t_d):
        # Project velocity and acceleration to tangent space
        vel_t = self.project_point_onto_plane(vel_t, x_t)  # just in case, velocity should already be tangent
        acc_t_d = self.project_point_onto_plane(acc_t_d, x_t)

        # Integrate acc in tangent space to velocity in tangent space
        vel_t_d = euler_integration(vel_t, acc_t_d, self.delta_t)

        # Compute exponential map
        delta_x_d_tangent = vel_t_d * self.delta_t
        x_t_d = self.exp_map_sphere(x_t, delta_x_d_tangent)
        return x_t_d, vel_t_d, acc_t_d

    def integrate_1st_order(self, x_t, vel_t_d):
        """
        Saturates and integrates state derivative for first-order systems
        """
        # Clip position (through the velocity)
        if self.saturate_transition and self.space == 'euclidean':
            max_vel_t_d = (1 - x_t) / self.delta_t
            min_vel_t_d = (-1 - x_t) / self.delta_t
            vel_t_d = torch.clamp(vel_t_d, min_vel_t_d, max_vel_t_d)

        # Integrate
        if self.space == 'euclidean':
            x_t_d = euler_integration(x_t, vel_t_d, self.delta_t)
        elif self.space == 'sphere':
            x_t_d, vel_t_d = self.integrate_non_euclidean_1st_order(x_t, vel_t_d)
        elif self.space == 'euclidean_sphere':
            # Integrate euclidean
            x_t_d_trans = euler_integration(x_t[:, :3], vel_t_d[:, :3], self.delta_t)

            # Integrate non-euclidean
            x_t_d_rot, vel_t_d_rot = self.integrate_non_euclidean_1st_order(x_t[:, 3:], vel_t_d[:, 3:])

            # Put together
            x_t_d = torch.cat([x_t_d_trans, x_t_d_rot], dim=1)
            vel_t_d = torch.cat([vel_t_d[:, :3], vel_t_d_rot], dim=1)

            # x_t_d = euler_integration(x_t, vel_t_d, self.delta_t)
            # projected_points = self.map_points_to_sphere(x_t_d[:, 3:])
            # x_t_d = torch.cat([x_t_d[:, :3], projected_points], dim=1)  # pytorch doesn't like inplace operations
        else:
            raise ValueError('Selected space not valid, options: euclidean, sphere and euclidean_sphere.')

        return x_t_d, vel_t_d

    def integrate_2nd_order(self, x_t, acc_t_d):
        """
        Saturates and integrates state derivative for second-order systems
        """
        # Separate state in position and velocity
        pos_t = x_t[:, :self.dim_position]
        vel_t = denormalize_state(x_t[:, self.dim_position:], self.min_vel, self.max_vel)

        # Clip position and velocity (through the acceleration)
        if self.saturate_transition and self.space == 'euclidean':
            # Position
            max_acc_t_d = (1 - pos_t - vel_t * self.delta_t) / self.delta_t**2
            min_acc_t_d = (-1 - pos_t - vel_t * self.delta_t) / self.delta_t**2
            acc_t_d = torch.clamp(acc_t_d, min_acc_t_d, max_acc_t_d)

            # Velocity
            max_acc_t_d = (self.max_vel - vel_t) / self.delta_t
            min_acc_t_d = (self.min_vel - vel_t) / self.delta_t
            acc_t_d = torch.clamp(acc_t_d, min_acc_t_d, max_acc_t_d)

        # Integrate
        if self.space == 'euclidean':
            vel_t_d = euler_integration(vel_t, acc_t_d, self.delta_t)
            pos_t_d = euler_integration(pos_t, vel_t_d, self.delta_t)
        elif self.space == 'sphere':
            pos_t_d, vel_t_d, acc_t_d = self.integrate_non_euclidean_2nd_order(pos_t, vel_t, acc_t_d)
        elif self.space == 'euclidean_sphere':
            # Integrate euclidean
            vel_t_d_trans = euler_integration(vel_t[:, :3], acc_t_d[:, :3], self.delta_t)
            pos_t_d_trans = euler_integration(pos_t[:, :3], vel_t_d_trans[:, :3], self.delta_t)

            # Integrate non-euclidean
            pos_t_d_rot, vel_t_d_rot, acc_t_d_rot = self.integrate_non_euclidean_2nd_order(pos_t[:, 3:], vel_t[:, 3:], acc_t_d[:, 3:])

            # Put together
            pos_t_d = torch.cat([pos_t_d_trans, pos_t_d_rot], dim=1)
            vel_t_d = torch.cat([vel_t_d_trans, vel_t_d_rot], dim=1)
            acc_t_d = torch.cat([acc_t_d[:, :3], acc_t_d_rot], dim=1)
        else:
            raise ValueError('Selected space not valid, options: euclidean, sphere and euclidean_sphere.')

        # Normalize velocity to have a state space between -1 and 1
        vel_t_d_norm = normalize_state(vel_t_d, self.min_vel, self.max_vel)

        # Create desired state
        x_t_d = torch.cat([pos_t_d, vel_t_d_norm], dim=1)

        return x_t_d, vel_t_d, acc_t_d

    def transition(self, x_t=None, **kwargs):
        """
        Computes dynamical system one-step transition
        """
        # If no state provided, assume perfect transition from previous desired state
        if x_t is None:
            x_t = self.x_t_d

        # Map task state to latent state (psi)
        self.y_t = self.model.encoder(x_t, self.primitive_type)

        # Map latent state to task state derivative (vel/acc) (phi)
        dx_t_d = self.map_to_derivative(self.y_t)

        # Saturate (to keep state inside boundary) and integrate derivative
        if self.order == 1:
            self.x_t_d, vel_t_d = self.integrate_1st_order(x_t, dx_t_d)
            acc_t_d = None  # no acceleration in first order systems
        elif self.order == 2:
            self.x_t_d, vel_t_d, acc_t_d = self.integrate_2nd_order(x_t, dx_t_d)
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        # Obstacle avoidance
        if 'obstacles' in kwargs:
            
            self.x_t_d, vel_t_d = self.obstacle_avoidance(x_t[:, :self.dim_position],
                                                          vel_t_d,
                                                          kwargs['obstacles'])
            if self.order == 2:
                self.x_t_d  = torch.cat([self.x_t_d, normalize_state(vel_t_d, self.min_vel, self.max_vel)], dim=1)

        # Collect transition info
        transition_info = {'desired state': self.x_t_d,
                           'desired velocity': vel_t_d,
                           'desired acceleration': acc_t_d,
                           'latent state': self.y_t}

        return transition_info

    def simulate(self, simulation_steps, **kwargs):
        """
        Simulates dynamical system
        """
        states_history = [self.x_t_d.cpu().detach().numpy()]
        with torch.no_grad():
            for t in range(simulation_steps - 1):
                # Do transition
                transition_info = self.transition(**kwargs)
                x_t = transition_info['desired state']

                # Append world transition
                states_history.append(x_t.cpu().detach().numpy())

        return np.array(states_history)

    def obstacle_avoidance(self, x_t, dx_t, obstacles):
        """
        Ellipsoidal multi-obstacle avoidance (paper: https://cs.stanford.edu/people/khansari/ObstacleAvoidance.html)
        """
        batch_size = dx_t.shape[0]
        n_obs = len(obstacles['centers'])

        # Reshape x
        x_t = x_t.view(batch_size, self.dim_position)

        # Denorm delta x
        delta_x = dx_t * self.delta_t  # delta x

        if not obstacles['centers']:
            # Integrate in time
            x_t_d = x_t + delta_x
            delta_x_mod = delta_x
        else:
            x_t = x_t.repeat_interleave(repeats=n_obs, dim=1).view(batch_size, self.dim_position, n_obs).transpose(1, 2)  # Repeat as many obstacles

            # Obstacles
            obs = torch.FloatTensor(normalize_state(np.array(obstacles['centers']),
                                                    x_min=self.x_min,
                                                    x_max=self.x_max)).repeat(batch_size, 1, 1).cuda()
            sf = torch.FloatTensor(obstacles['safety_margins']).repeat(batch_size, 1, 1).cuda()

            a = torch.FloatTensor(get_derivative_normalized_state(np.array(obstacles['axes']),
                                                                  x_min=self.x_min,
                                                                  x_max=self.x_max)).repeat(batch_size, 1, 1).cuda()

            # Get modulation Ellipsoid
            x_ell = x_t - obs

            # Get Gamma
            a = a * sf
            Gamma = torch.sum((x_ell / a)**2, dim=2)  # TODO: include p here, now p=1

            Gamma[Gamma < 1] = 1e3  # If inside obstacle, ignore obstacle

            # Get weights
            Gamma_k = Gamma.view(batch_size, n_obs, 1).repeat(1, 1, n_obs)
            Gamma_i = Gamma.repeat_interleave(repeats=n_obs, dim=1).view(batch_size, n_obs, n_obs).transpose(1, 2)
            filter_i_eq_k = 1e30 * torch.eye(n_obs).repeat(batch_size, 1, 1).cuda() + torch.ones(n_obs).repeat(batch_size, 1, 1).cuda()  # Values to ignore when gamma i=k
            Gamma_i = filter_i_eq_k * Gamma_i  # Apply filter
            w = torch.prod((Gamma_i - 1) / ((Gamma_k - 1) + (Gamma_i - 1)), dim=2)  # Compute w

            # Get basis matrix
            nv = (2 / a) * (x_ell / a)  # TODO: extend to p > 1
            E = torch.zeros(batch_size, n_obs, self.dim_position, self.dim_position).cuda()
            E[:, :, :, 0] = nv
            E[:, :, 0, 1:self.dim_position] = nv[:, :, 1:self.dim_position]

            I = torch.eye(self.dim_position-1).repeat(batch_size * n_obs, 1, 1).cuda()
            e_last = nv.view(batch_size * n_obs, self.dim_position)[:, 0].view(batch_size * n_obs, 1)[:, :, None]
            E[:, :, 1:self.dim_position, 1:self.dim_position] = (- I * e_last).view(batch_size, n_obs, self.dim_position-1, self.dim_position-1)

            D = torch.zeros(batch_size, n_obs, self.dim_position, self.dim_position).cuda()

            D[:, :, 0, 0] = 1 - (w / Gamma)

            for i in range(self.dim_position-1):
                D[:, :, i + 1, i + 1] = 1 + (w / Gamma)

            # Get modulation matrix
            E = E.view(batch_size * n_obs, self.dim_position, self.dim_position)
            D = D.view(batch_size * n_obs, self.dim_position, self.dim_position)
            M = torch.bmm(torch.bmm(E, D), torch.inverse(E))  # EDE^{-1}
            M = M.view(batch_size, n_obs, self.dim_position, self.dim_position)

            # Modulate DS
            delta_x_mod = delta_x.view(batch_size, self.dim_position, 1)

            for i in range(n_obs):  # TODO: doable without for?
                delta_x_mod = torch.bmm(M[:, i, :, :], delta_x_mod)
            delta_x_mod = delta_x_mod.view(batch_size, self.dim_position)

            # Integrate in time
            x_t_d = x_t[:, 0, :] + delta_x_mod

        # Clamp values inside workspace
        x_t_d = torch.clamp(x_t_d, -1, 1)

        return x_t_d, delta_x_mod / self.delta_t