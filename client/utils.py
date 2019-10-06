def cal_pts(im):
    import numpy as np

    KPOINTS = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32 )

    E0PX, E0PY = KPOINTS[0]
    E1PX, E1PY = KPOINTS[1]
    NPX, NPY = KPOINTS[2]
    M0PX, M0PY = KPOINTS[3]
    M1PX, M1PY = KPOINTS[4]

    MOUTH_W = M1PX - M0PX
    MOUTH_W *= 1.236
    MOUTH_BXC = (M0PX + M1PX) * 0.5
    MOUTH_BYC = (M0PY + M1PY) * 0.5
    MOUTH_H = MOUTH_BYC - NPY

    MOUTH_BX0, MOUTH_BX1 = MOUTH_BXC - 0.5 * MOUTH_W, MOUTH_BXC + 0.5 * MOUTH_W
    MOUTH_BY0, MOUTH_BY1 = MOUTH_BYC - 0.5 * MOUTH_H, MOUTH_BYC + 0.5 * MOUTH_H

    EYE_W = (E1PX - E0PX) * 0.5
    EYE_W *= 1.236
    EYE_H = MOUTH_H * 0.618
    EYE0_W = NPX - E0PX
    EYE0_W = (EYE0_W + EYE_W) * 0.5
    EYE1_W = E1PX - NPX
    EYE1_W = (EYE1_W + EYE_W) * 0.5
    EYE0_H = EYE_H * EYE0_W / EYE_W
    EYE1_H = EYE_H * EYE1_W / EYE_W

    EYE0_BX0, EYE0_BX1 = E0PX - 0.5 * EYE0_W, E0PX + 0.5 * EYE0_W
    EYE0_BY0, EYE0_BY1 = E0PY - 0.5 * EYE0_H, E0PY + 0.5 * EYE0_H
    EYE1_BX0, EYE1_BX1 = E1PX - 0.5 * EYE1_W, E1PX + 0.5 * EYE1_W
    EYE1_BY0, EYE1_BY1 = E1PY - 0.5 * EYE1_H, E1PY + 0.5 * EYE1_H

    NOSE_BX0, NOSE_BX1 = NPX - 0.618 * EYE1_W, NPX + 0.618 * EYE0_W
    NOSE_BY1 = MOUTH_BY0
    NOSE_BY0 = min(EYE0_BY1, EYE1_BY1)

    MOUTH_BX0 = np.round(MOUTH_BX0).astype(np.int)
    MOUTH_BX1 = np.round(MOUTH_BX1).astype(np.int)
    MOUTH_BY0 = np.round(MOUTH_BY0).astype(np.int)
    MOUTH_BY1 = np.round(MOUTH_BY1).astype(np.int)
    NOSE_BX0 = np.round(NOSE_BX0).astype(np.int)
    NOSE_BX1 = np.round(NOSE_BX1).astype(np.int)
    NOSE_BY0 = np.round(NOSE_BY0).astype(np.int)
    NOSE_BY1 = np.round(NOSE_BY1).astype(np.int)
    EYE0_BX0 = np.round(EYE0_BX0).astype(np.int)
    EYE0_BX1 = np.round(EYE0_BX1).astype(np.int)
    EYE0_BY0 = np.round(EYE0_BY0).astype(np.int)
    EYE0_BY1 = np.round(EYE0_BY1).astype(np.int)
    EYE1_BX0 = np.round(EYE1_BX0).astype(np.int)
    EYE1_BX1 = np.round(EYE1_BX1).astype(np.int)
    EYE1_BY0 = np.round(EYE1_BY0).astype(np.int)
    EYE1_BY1 = np.round(EYE1_BY1).astype(np.int)

    EYE_BX0 = min(EYE0_BX0, EYE1_BX0)
    EYE_BX1 = max(EYE0_BX1, EYE1_BX1)
    EYE_BY0 = min(EYE0_BY0, EYE1_BY0)
    EYE_BY1 = max(EYE0_BY1, EYE1_BY1)

    return ((EYE_BX0, EYE_BY0), (EYE_BX1, EYE_BY1)), \
            ((NOSE_BX0, NOSE_BY0), (NOSE_BX1, NOSE_BY1)), \
            ((MOUTH_BX0, MOUTH_BY0), (MOUTH_BX1, MOUTH_BY1))

(EYE_PT0, EYE_PT1), (NOSE_PT0, NOSE_PT1), (MOUTH_PT0, MOUTH_PT1) = cal_pts(None)
