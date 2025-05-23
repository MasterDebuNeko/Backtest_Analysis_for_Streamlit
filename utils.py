# utils.py

import numpy as np
import pandas as pd # แม้ตอนนี้อาจจะยังไม่ได้ใช้โดยตรงในทุกฟังก์ชัน แต่เผื่ออนาคต
from matplotlib.colors import Normalize # สำหรับ CustomDivergingNorm

def clean_number(val):
    """Convert string with commas/spaces to float. Return NaN if fails."""
    try:
        return float(str(val).replace(',', '').replace(' ', ''))
    except Exception:
        return np.nan

def validate_stop_loss(stop_loss_pct):
    """
    Ensure stop_loss_pct is a float between 0 and 1 (not inclusive).
    Raise ValueError if not valid.
    """
    try:
        pct = float(stop_loss_pct)
        if not (0 < pct < 1):
            raise ValueError("stop_loss_pct ต้องเป็นตัวเลขทศนิยมที่มากกว่า 0 และน้อยกว่า 1 เช่น 0.002 (สำหรับ 0.2%)")
        return pct
    except Exception:
        raise ValueError("stop_loss_pct ต้องเป็นตัวเลขทศนิยมที่มากกว่า 0 และน้อยกว่า 1 เช่น 0.002 (สำหรับ 0.2%)")

def safe_divide(numerator, denominator):
    """Elementwise safe division: if denom is 0 or NaN, return NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((denominator == 0) | pd.isnull(denominator) | (denominator == np.inf) | (denominator == -np.inf),
                          np.nan,
                          numerator / denominator)
    return result

class CustomDivergingNorm(Normalize):
    """
    Normalize that maps vcenter=0 to white in colormap.
    Negative values to red, positive values to blue.
    """
    def __init__(self, vmin=None, vmax=None, vcenter=0, clip=False): # Added default vmin, vmax
        super().__init__(vmin, vmax, clip)
        self.vcenter = vcenter

    def __call__(self, value, clip=None):
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        
        if vmin is None or vmax is None or vmin == vmax:
            return np.ma.masked_array(np.full_like(value, 0.5, dtype=float))

        value = np.ma.masked_array(value, np.isnan(value))
        result = np.ma.masked_array(np.zeros_like(value, dtype=float), value.mask)
        
        neg_mask = value < vcenter
        if vcenter > vmin:
            result[neg_mask] = 0.5 * (value[neg_mask] - vmin) / (vcenter - vmin)
        elif vmin == vcenter:
             result[neg_mask] = 0.0

        pos_mask = value >= vcenter
        if vmax > vcenter:
            result[pos_mask] = 0.5 + 0.5 * (value[pos_mask] - vcenter) / (vmax - vcenter)
        elif vmax == vcenter:
            result[pos_mask] = 0.5
            if vmin == vmax:
                 result[pos_mask] = 0.5

        if self.clip:
            result = np.ma.clip(result, 0, 1)
            
        return result