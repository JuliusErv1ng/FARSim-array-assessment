"""Time-domain waveform synthesis layer for FARSim."""

from .api import ArrayTriggerConfig, ApertureComparison, FARSimAPI, FARSimAPIConfig, PrimaryParticle, default_l1_config
from .predictor import FARSimWaveformPredictor, StationPrediction, synthesize_trace_at_position
from .aperture_modes import (
    AdaptiveVoltageMap,
    ApertureModeResult,
    NoiseVoltageLibrary,
    VoltageMap,
    build_adc_noise_library,
    compute_adaptive_l1_probability_map,
    compute_adaptive_voltage_peak_map,
    compute_l1_probability_map,
    compute_voltage_peak_map,
    field_threshold_aperture,
    poisson_binomial_at_least,
    probability_map_array_efficiency,
    voltage_l1_aperture,
    voltage_threshold_aperture,
)
from .complex_template import ComplexTemplateModel
from .digital_filters import ADCDigitalFilterConfig, adc_filter_metadata, apply_adc_iir_fir
from .frequency_template import FrequencyTemplateModel
from .kernel_template import KernelTemplateModel
from .pca_template import PCATemplateModel
from .mlp_pca_template import MLPPcaTemplateModel
from .rfchain_adapter import RFChainLengthNAdapter, RFChainVoltageResult
from .synthesizer import WaveformInput, WaveformSynthesizer
from .templates import AsymmetricGaussianTailParams, asymmetric_gaussian_tail, bipolar_gaussian_derivative
from .voltage_trigger import AERAVoltageTriggerConfig, VoltageTriggerDecision, aera_voltage_trigger
from .width_model import estimate_fwhm_ns

__all__ = [
    "AERAVoltageTriggerConfig",
    "ADCDigitalFilterConfig",
    "AdaptiveVoltageMap",
    "ApertureComparison",
    "ApertureModeResult",
    "ArrayTriggerConfig",
    "AsymmetricGaussianTailParams",
    "ComplexTemplateModel",
    "FARSimAPI",
    "FARSimAPIConfig",
    "FARSimWaveformPredictor",
    "FrequencyTemplateModel",
    "KernelTemplateModel",
    "PCATemplateModel",
    "MLPPcaTemplateModel",
    "NoiseVoltageLibrary",
    "PrimaryParticle",
    "RFChainLengthNAdapter",
    "RFChainVoltageResult",
    "StationPrediction",
    "VoltageMap",
    "VoltageTriggerDecision",
    "WaveformInput",
    "WaveformSynthesizer",
    "aera_voltage_trigger",
    "adc_filter_metadata",
    "apply_adc_iir_fir",
    "asymmetric_gaussian_tail",
    "bipolar_gaussian_derivative",
    "build_adc_noise_library",
    "compute_adaptive_l1_probability_map",
    "compute_l1_probability_map",
    "compute_adaptive_voltage_peak_map",
    "compute_voltage_peak_map",
    "default_l1_config",
    "estimate_fwhm_ns",
    "field_threshold_aperture",
    "poisson_binomial_at_least",
    "probability_map_array_efficiency",
    "synthesize_trace_at_position",
    "voltage_l1_aperture",
    "voltage_threshold_aperture",
]
