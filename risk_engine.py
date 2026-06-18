def expected_loss(models, pd, emi, utilization, income,
                  salary_flag, util_flag):

    config = models["config"]

    # Navigate correctly
    el_engine = config["expected_loss_engine"]

    #  EAD RULES 
    emi_mult = el_engine["ead_rules"]["emi_multiplier"]
    util_mult = el_engine["ead_rules"]["utilization_income_multiplier"]

    ead = emi * emi_mult + utilization * income * util_mult

    #  LGD RULES 
    base_lgd = el_engine["lgd_rules"]["base_lgd"]
    salary_add = el_engine["lgd_rules"]["salary_delay_add"]
    util_add = el_engine["lgd_rules"]["utilization_add"]

    lgd = base_lgd + salary_add * salary_flag + util_add * util_flag

    #  EXPECTED LOSS 
    el = pd * lgd * ead

    return el, lgd, ead


def risk_bucket(pd):

    if pd >= 0.60:
        return "VERY HIGH"
    elif pd >= 0.35:
        return "HIGH"
    elif pd >= 0.15:
        return "MEDIUM"
    elif pd >= 0.05:
        return "LOW"
    else:
        return "VERY LOW"