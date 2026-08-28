"""
Bueorm Core - Open Architectural Registry
Enables dynamic registration and extensible discovery of architectures,
layers, attention mechanisms, backbones, and expert modules.
"""

from typing import Dict, Any, Type, Callable, Optional
import inspect


class Registry:
    """Generic module and component registry."""
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        def decorator(obj: Any) -> Any:
            reg_name = name if name is not None else obj.__name__
            if reg_name in self._registry:
                # Allow re-registering or updating
                pass
            self._registry[reg_name.lower()] = obj
            self._registry[reg_name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name in self._registry:
            return self._registry[name]
        lower_name = name.lower()
        if lower_name in self._registry:
            return self._registry[lower_name]
        available = list(self._registry.keys())
        raise KeyError(f"'{name}' not found in registry '{self.name}'. Available: {available}")

    def list_available(self) -> list:
        return sorted(list(set(self._registry.keys())))

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name.lower() in self._registry


# Global Registries
MODEL_REGISTRY = Registry("Models")
LAYER_REGISTRY = Registry("Layers")
BACKBONE_REGISTRY = Registry("Backbones")
ROUTER_REGISTRY = Registry("Routers")


def register_model(name: Optional[str] = None) -> Callable:
    """Decorator to register a new neural network model architecture."""
    return MODEL_REGISTRY.register(name)


def register_layer(name: Optional[str] = None) -> Callable:
    """Decorator to register a new layer/block module."""
    return LAYER_REGISTRY.register(name)


def register_backbone(name: Optional[str] = None) -> Callable:
    """Decorator to register a vision or language backbone."""
    return BACKBONE_REGISTRY.register(name)


def register_router(name: Optional[str] = None) -> Callable:
    """Decorator to register an MoE router strategy."""
    return ROUTER_REGISTRY.register(name)
