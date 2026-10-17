from wave_sim2d.wave_simulation import SceneObject
import torch
import torch.nn.functional as F
import numpy as np

class StrainRefractiveIndex(SceneObject):
    """
    Implements a dynamic refractive index field that linearly depends on the strain of the current field.
    The refractive index within the entire domain is overwritten
    """

    def __init__(self, refractive_index_offset, coupling_constant):
        """
        Creates a strain refractive index field object
        :param coupling_constant: coupling constant between the strain and the refractive index
        """
        self.device = torch.device("cpu")  # simulator.bind owns device selection
        self.coupling_constant = coupling_constant
        self.refractive_index_offset = refractive_index_offset

        # Define derivative kernels for computing strain
        self.du_dx_kernel = torch.tensor([[-1, 0.0, 1]], dtype=torch.float32, device=self.device)
        self.du_dx_kernel = self.du_dx_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, 3)
        
        self.du_dy_kernel = torch.tensor([[-1], [0.0], [1]], dtype=torch.float32, device=self.device)
        self.du_dy_kernel = self.du_dy_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 1)

        self.strain_field = None

    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        # compute strain using convolution
        field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0), mode='constant', value=0.0)
        du_dx = F.conv2d(field_padded, self.du_dx_kernel, padding=0).squeeze(0).squeeze(0)
        
        field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode='constant', value=0.0)
        du_dy = F.conv2d(field_padded, self.du_dy_kernel, padding=0).squeeze(0).squeeze(0)

        self.strain_field = torch.sqrt(du_dx**2 + du_dy**2)

        # compute refractive index from strain
        refractive_index_field = self.refractive_index_offset + self.strain_field * self.coupling_constant

        # assign wave speed using refractive index from above
        wave_speed_field[:] = 1.0 / torch.clamp(refractive_index_field, 0.9, 10.0)

    def update_field(self, field: torch.Tensor, t):
        pass

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass
