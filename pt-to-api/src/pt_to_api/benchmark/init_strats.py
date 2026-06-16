from dataclasses import dataclass


@dataclass
class StandardInitStrategy:
    def __repr__(self):
        return "'StandardInitStrategy'"


@dataclass
class ZeroInitStrategy:
    def __repr__(self):
        return "'ZeroInitStrategy'"


@dataclass
class NoInitStrategy:
    def __repr__(self):
        return "'NoInitStrategy'"


@dataclass
class SvdInitStrategy:
    def __repr__(self):
        return "'SvdInitStrategy'"


@dataclass
class IcaInitStrategy:
    iters: int

    def __repr__(self):
        return f"'IcaInitStrategy({self.iters})'"


@dataclass
class WarmupInitStrategy:
    warmup_epochs: int

    def __repr__(self):
        return f"'WarmupInitStrategy({self.warmup_epochs})'"


@dataclass
class OnlyInitEncoderStrategy:
    def __repr__(self):
        return "'OnlyInitEncoderStrategy'"


InitStrategy = (
    SvdInitStrategy
    | WarmupInitStrategy
    | StandardInitStrategy
    | IcaInitStrategy
    | NoInitStrategy
    | OnlyInitEncoderStrategy
)
