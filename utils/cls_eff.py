# Classification function
def classify_efficiency(eff):
    if eff <= 0.4:
        return 0
    elif eff <= 0.8:
        return 1
    else:
        return 2

