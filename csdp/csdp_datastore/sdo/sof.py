from .sdo_base import SleepdataOrg
from ..models import TTRef, Mapping

class SOF(SleepdataOrg):
    
    def channel_mapping(self):
        return {
            "C3": Mapping(TTRef.C3, TTRef.Fpz),
            "C4": Mapping(TTRef.C4, TTRef.Fpz),
            "A1": Mapping(TTRef.LPA, TTRef.Fpz),
            "A2": Mapping(TTRef.RPA, TTRef.Fpz),
            "ROC": Mapping(TTRef.ER, TTRef.Fpz),
            "LOC": Mapping(TTRef.EL, TTRef.Fpz)
        }
        