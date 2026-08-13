"""hoa64 — spatial calculator: 7th-order Ambisonics (Ambix ACN / SN3D).

Fixed spherical-harmonic geometry (Farina / Ambix), not a learned model.
State dimension is exactly 64 = (7+1)**2.

Phases:
  0 — basis, encode/decode, rotate, intensity DOA, RNN pose stub
  1 — WAV I/O, HOA streams, STFT bands, JSON spatial reports for other AIs
  2 — Fast Wigner-D HOA rotation (default)
  3 — Vision boxes/rays on sphere + A/V fuse + Qwythos API
"""

from .basis import (
    MAX_ORDER,
    N_CHANNELS,
    acn_index,
    acn_nm,
    channel_names,
    sh_sn3d,
    sh_sn3d_batch,
    unit_vector,
    az_el_from_unit,
)
from .encode import encode_points, encode_plane_waves, mix
from .decode import decode_directions, decode_grid, beamform
from .rotate import rotate_yaw_pitch_roll, rotate_matrix_order1
from .wigner import hoa_rotation_matrix, apply_hoa_rotation, rotation_matrix_zyx
from .analysis import (
    field_energy,
    intensity_vector,
    doa_from_intensity,
    directional_power,
    peak_direction,
)
from .report import (
    SpatialReport,
    report_from_hoa,
    report_from_mono_wav,
    report_from_ambix_wav,
    report_from_scene,
)
from .stream import SourceSpec, encode_mono_plane_wave, encode_scene, analyze_hoa_frames
from .vision import report_from_boxes, fuse_reports, encode_boxes_to_hoa
from .detector import detections_to_sphere_boxes, detect_to_sphere
from .conditioning import build_conditioning, panner_report, spatial_prompt_fragment
from . import hadamard
from .hadamard import (
    check as hadamard_check,
    verify as hadamard_verify,
    normalize as hadamard_normalize,
    modular_check,
    sylvester,
    paley,
    hadamard_known,
    hadamard_orders,
    random_seed as hadamard_random_seed,
    local_search as hadamard_local_search,
    ils_search,
    det_log10 as hadamard_det_log10,
    load_modular_seed,
)

__all__ = [
    "MAX_ORDER",
    "N_CHANNELS",
    "acn_index",
    "acn_nm",
    "channel_names",
    "sh_sn3d",
    "sh_sn3d_batch",
    "unit_vector",
    "az_el_from_unit",
    "encode_points",
    "encode_plane_waves",
    "mix",
    "decode_directions",
    "decode_grid",
    "beamform",
    "rotate_yaw_pitch_roll",
    "rotate_matrix_order1",
    "hoa_rotation_matrix",
    "apply_hoa_rotation",
    "rotation_matrix_zyx",
    "field_energy",
    "intensity_vector",
    "doa_from_intensity",
    "directional_power",
    "peak_direction",
    "SpatialReport",
    "report_from_hoa",
    "report_from_mono_wav",
    "report_from_ambix_wav",
    "report_from_scene",
    "SourceSpec",
    "encode_mono_plane_wave",
    "encode_scene",
    "analyze_hoa_frames",
    "report_from_boxes",
    "fuse_reports",
    "encode_boxes_to_hoa",
    "detections_to_sphere_boxes",
    "detect_to_sphere",
    "build_conditioning",
    "panner_report",
    "spatial_prompt_fragment",
    "hadamard",
    "hadamard_check",
    "hadamard_verify",
    "hadamard_normalize",
    "modular_check",
    "sylvester",
    "paley",
    "hadamard_known",
    "hadamard_orders",
    "hadamard_random_seed",
    "hadamard_local_search",
    "ils_search",
    "hadamard_det_log10",
    "load_modular_seed",
]

__version__ = "0.5.0"
