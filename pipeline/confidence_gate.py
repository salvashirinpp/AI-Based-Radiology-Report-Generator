HIGH = 0.75
LOW = 0.40

def confidence_gate(prob_dict):
    findings = []
    indeterminate = []
    normal = []

    for label, prob in prob_dict.items():
        if prob >= HIGH:
            findings.append(label)
        elif prob >= LOW:
            indeterminate.append(label)

    if not findings and not indeterminate:
        normal.append("no acute cardiopulmonary abnormality")

    return {
        "positive_findings": findings,
        "indeterminate_findings": indeterminate,
        "normal": normal
    }
