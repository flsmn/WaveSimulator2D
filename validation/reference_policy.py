"""Validation-only policies and diagnostics; independent of the runtime solver."""
import json
import math
from pathlib import Path


def load_policy(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if 'extends' not in data:
        return data
    result = load_policy(path.parent/data['extends'])
    for group, values in data['overrides'].items():
        result[group] = {**result[group], **values}
    result['profile'] = {k: v for k, v in data.items() if k not in ('extends', 'overrides')}
    return result


def snell_diagnostics(incident_degrees, transmitted_degrees, *, n1=1., n2=1.5,
                      nominal_incident_degrees=25.):
    """Report nominal and measured-incidence references without changing v1 PASS."""
    def reference(angle):
        sine = n1/n2*math.sin(math.radians(angle))
        if abs(sine) > 1:
            raise ValueError('No transmitted propagating plane wave (total internal reflection).')
        return math.degrees(math.asin(sine))
    nominal, conditioned = reference(nominal_incident_degrees), reference(incident_degrees)
    return {
        'nominal_incidence_degrees': nominal_incident_degrees,
        'measured_incidence_degrees': incident_degrees,
        'transmission_degrees': transmitted_degrees,
        'nominal_snell_reference_degrees': nominal,
        'nominal_angle_error_degrees': abs(transmitted_degrees-nominal),
        'measured_incidence_snell_reference_degrees': conditioned,
        'measured_incidence_angle_error_degrees': abs(transmitted_degrees-conditioned),
        'incidence_estimation_deviation_degrees': abs(incident_degrees-nominal_incident_degrees),
        'quantitative_angle_accepted': False,
        'note': 'Conditional diagnostic only: finite illumination, fit and interface errors are not separated.',
    }
