from wave_sim2d.wave_simulation import SceneObject
import torch
import numpy as np


class StaticDampening(SceneObject):
    """
    Implements a static dampening field that overwrites the entire domain.
    Therefore, us this as base layer in your scene.
    """

    def __init__(self, dampening_field, border_thickness):
        """
        Creates a static dampening field object
        @param dampening_field: A NxM array with dampening factors (1.0 equals no dampening) of the same size as the simulation domain.
        @param pml_thickness: Thickness of the Perfectly Matched Layer (PML) at the boundaries to prevent reflections.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        w = dampening_field.shape[1]
        h = dampening_field.shape[0]
        self.d = torch.ones((h, w), dtype=torch.float32, device=self.device)
        
        # Convert input to tensor if needed
        if not isinstance(dampening_field, torch.Tensor):
            dampening_field = torch.from_numpy(dampening_field).float()
        dampening_field = dampening_field.to(self.device)
        
        self.d = torch.clamp(dampening_field, 0.0, 1.0)

        # apply border dampening
        for i in range(border_thickness):
            v = (i / border_thickness) ** 0.5
            self.d[i, i:w - i] = v
            self.d[-(1 + i), i:w - i] = v
            self.d[i:h - i, i] = v
            self.d[i:h - i, -(1 + i)] = v

    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        assert (dampening_field.shape == self.d.shape)

        # overwrite existing dampening field
        dampening_field[:] = self.d

    def update_field(self, field: torch.Tensor, t):
        pass

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass
