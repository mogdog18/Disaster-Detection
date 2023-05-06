from .disaster_sam import DisasterSamV1, DisasterSamV2


class ModelRegistry:
    DisasterSamV1 = DisasterSamV1
    DisasterSamV2 = DisasterSamV2

    @classmethod
    def get_model(cls, name):
        return getattr(cls, name)
