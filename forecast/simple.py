def ans_static(info)->float:
    return info["First 6 mo. Avg. GAS (Mcf)"]

def ans_serial(info)->float:
    return info["Last 6 mo. Avg. GAS (Mcf)"]

def test_static(info)->float:
    return info["MD (All Wells) (ft)"]*10

def test_serial(info)->float:
    return info["GAS_MONTH_9"]
