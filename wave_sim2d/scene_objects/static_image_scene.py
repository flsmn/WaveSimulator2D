from wave_sim2d.wave_simulation import SceneObject

import numpy as np
import torch
from wave_sim2d.scene_objects.static_dampening import StaticDampening
from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex


class StaticImageScene(SceneObject):
    """
    Implements static scene, where the RGB channels of the input image encode the refractive index, the dampening and sources.
    This class allows to use an image editor to create scenes.
    """
    def __init__(self, scene_image, source_amplitude=1.0, source_fequency_scale=1.0):
        """
        load source from an image description
        The simulation scenes are given as an 8Bit RGB image with the following channel semantics:
            * Red:   The Refractive index times 100 (for refractive index 1.5 you would use value 150)
            * Green: Each pixel with a green value above 0 is a sinusoidal wave source. The green value
                     defines its frequency. WARNING: Do not use antialiasing for the green channel !
            * Blue:  Absorbtion field. Larger values correspond to higher dampening of the waves,
                     use graduated transitions to avoid reflections
        """
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # Set the opacity of source pixels to incoming waves. If the opacity is 0.0
        # the field will be completely overwritten by the source term
        # a nonzero value (e.g 0.5) allows for antialiasing of sources to work
        self.source_opacity = 0.9

        # set refractive index field
        self.refractive_index = StaticRefractiveIndex(scene_image[:, :, 0] / 100)

        # set absorber field
        self.dampening = StaticDampening(1.0 - scene_image[:, :, 2] / 255, border_thickness=48)

        # set sources, each entry describes a source with the following parameters:
        # (x, y, phase, amplitude, frequency)
        sources_pos = np.flip(np.argwhere(scene_image[:, :, 1] > 0), axis=1)
        phase_amplitude_freq = np.tile(np.array([0, source_amplitude, 0.3]), (sources_pos.shape[0], 1))
        self.sources = np.concatenate((sources_pos, phase_amplitude_freq), axis=1)

        # set source frequency to channel value
        self.sources[:, 4] = scene_image[sources_pos[:, 1], sources_pos[:, 0], 1] / 255 * 0.5 * source_fequency_scale
        self.sources = torch.from_numpy(self.sources).to(torch.float32).to(self.device)

    def render(self, field: torch.Tensor, wave_speed_field: torch.Tensor, dampening_field: torch.Tensor):
        """
        render the stat
        """
        self.dampening.render(field, wave_speed_field, dampening_field)
        self.refractive_index.render(field, wave_speed_field, dampening_field)

    def update_field(self, field: torch.Tensor, t):
        # Update the sources in the simulation field based on their properties.
        v = torch.sin(self.sources[:, 2] + self.sources[:, 4] * t) * self.sources[:, 3]
        coords = self.sources[:, 0:2].to(torch.int64)

        o = self.source_opacity
        field[coords[:, 1], coords[:, 0]] = field[coords[:, 1], coords[:, 0]] * o + v * (1.0 - o)

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass
