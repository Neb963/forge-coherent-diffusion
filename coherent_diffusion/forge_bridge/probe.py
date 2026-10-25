"""Read-only introspection. Does not load models, modify hooks or inspect prompt text."""
from __future__ import annotations
from typing import Any

UPSTREAM_INSPECTED = "Haoming02/sd-webui-forge-classic@0c388f8b26d26d475c2c7104aea4eafdf7bbb3e7"

def _type(obj: Any) -> str | None:
    if obj is None: return None
    c=type(obj)
    return f"{c.__module__}.{c.__qualname__}"

def _attr(obj: Any, name: str, default=None) -> Any:
    if obj is None: return default
    try: return getattr(obj,name,default)
    except Exception: return default

def _bool(obj: Any, name: str) -> bool | None:
    x=_attr(obj,name)
    return x if type(x) is bool else None

def _encoder_evidence(clip: Any) -> dict:
    """No parameter values cloned/transferred; a bounded metadata scan only."""
    result={"patcher_type":_type(_attr(clip,"patcher")),
            "cond_stage_model_type":_type(_attr(clip,"cond_stage_model")),
            "gguf_parameter_marker_seen":False,
            "gguf_parameter_scan":"not_attempted",
            "quantization":"unverified", "actual_encode":"not_tested"}
    model=_attr(clip,"cond_stage_model")
    params=_attr(model,"named_parameters")
    if not callable(params):
        return result
    result["gguf_parameter_scan"]="up_to_64_parameter_headers"
    try:
        for i,(_,p) in enumerate(params()):
            if i>=64: break
            if _attr(p,"gguf_cls") is not None:
                result["gguf_parameter_marker_seen"]=True
                break
    except Exception:
        result["gguf_parameter_scan"]="unavailable"
    # Even an observed GGUF marker does not prove Q4 or a successful Anima encode.
    return result


def _unet_hooks(unet: Any) -> dict:
    """Names and booleans only; never export weights, tensor values or prompts."""
    if unet is None:
        return {"available": False}
    options = _attr(unet, "model_options")
    keys = sorted(str(k)[:64] for k in options)[:24] if isinstance(options, dict) else []
    nested = options.get("transformer_options") if isinstance(options, dict) else None
    nested_keys = (sorted(str(k)[:64] for k in nested)[:12]
                   if isinstance(nested, dict) else [])
    return {
        "available": True,
        "controlnet_attached": _attr(unet, "controlnet_linked_list") is not None,
        "extra_concat_condition_attached": _attr(unet, "extra_concat_condition") is not None,
        "model_options_type": type(options).__name__,
        "model_option_keys": keys,
        "transformer_option_keys": nested_keys,
        "keys_truncated": isinstance(options, dict) and len(options) > 24,
    }

def _inspect(model: Any, *, source: str) -> dict[str,Any]:
    obj=_attr(model,"forge_objects")
    unet=_attr(obj,"unet")
    clip=_attr(obj,"clip")
    vae=_attr(obj,"vae")
    config=_attr(model,"model_config")
    engine=_type(model)
    if _bool(model,"is_sdxl") is True:
        family="SDXL (host flag)"
    elif engine and engine.endswith(".Anima"):
        family="Anima (class evidence)"
    else:
        family="unverified"
    return {
        "probe_version":1,
        "source":source,
        "upstream_reference_inspected_not_runtime_commit":UPSTREAM_INSPECTED,
        "runtime_commit":"unverified",
        "model_loaded":model is not None and engine != "modules.sd_models.FakeInitialModel",
        "model_type":engine,
        "model_family":family,
        "model_config_type":_type(config),
        "model_type_name":_attr(_attr(config,"model_type"),"name"),
        "latent_format_type":_type(_attr(config,"latent_format")),
        "unet_patcher_type":_type(unet),
        "vae_wrapper_type":_type(vae),
        "unet_hooks":_unet_hooks(unet),
        "is_sdxl":_bool(model,"is_sdxl"),
        "is_wan":_bool(model,"is_wan"),
        "has_conditioning_entry":callable(_attr(model,"get_learned_conditioning")),
        "has_decode_entry":callable(_attr(model,"decode_first_stage")),
        "text_encoder":_encoder_evidence(clip),
        "capabilities":{
            "positional_global_coordinates":"unverified",
            "solver_prediction_parameterization":"unverified",
            "synchronized_tiled_backend":"not_implemented_P01",
            "regional_attention":"not_implemented_P01",
            "regional_lora":"not_implemented_P01",
            "controlnet":"unverified",
            "sampler_step_sigma":"not_observed_without_sampling",
            "q4_gguf_anima_encode":"not_tested",
            "post_failure_cleanup":"no_owned_hooks_or_patches",
        }
    }

def inspect_loaded_host() -> dict[str,Any]:
    """Inspect only an already resident host model; never ask Forge to load one."""
    try:
        from modules import sd_models
        model_data=_attr(sd_models,"model_data")
        model=_attr(model_data,"sd_model")
        return _inspect(model,source="sd_models.model_data.sd_model")
    except ImportError:
        return _inspect(None,source="Forge modules unavailable (offline)")

def inspect_process(p: Any) -> dict[str,Any]:
    """Observation from host process callback. No mutation of p or global objects."""
    model=_attr(p,"sd_model")
    result=_inspect(model,source="Script.process(p)")
    result["sampling_request"]={
        "sampler_name":_attr(p,"sampler_name"),
        "steps":_attr(p,"steps"),
        "width":_attr(p,"width"),
        "height":_attr(p,"height"),
        # Model prediction sigma is not accessible from this entry point.
        "sigma_sequence":"not_observed",
    }
    return result
