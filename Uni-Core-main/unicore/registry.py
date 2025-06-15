import argparse

REGISTRIES = {}


def setup_registry(registry_name: str, base_class=None, default=None):
    assert registry_name.startswith("--")
    registry_name = registry_name[2:].replace("-", "_")
    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()
    if registry_name in REGISTRIES:
        return
    REGISTRIES[registry_name] = {"registry": REGISTRY, "default": default}

    def build_x(args, *extra_args, **extra_kwargs):
        choice = getattr(args, registry_name, None)
        if choice is None:
            return None
        cls = REGISTRY[choice]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls
        set_defaults(args, cls)
        return builder(args, *extra_args, **extra_kwargs)

    def register_x(name):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate {} ({})".format(registry_name, name)
                )
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    "Cannot register {} with duplicate class name ({})".format(
                        registry_name, cls.__name__
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError(
                    "{} must extend {}".format(cls.__name__, base_class.__name__)
                )
            REGISTRY[name] = cls
            REGISTRY_CLASS_NAMES.add(cls.__name__)
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY


def set_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)
