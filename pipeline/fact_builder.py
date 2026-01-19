def build_facts(confidence_output):
    return {
        "findings": confidence_output["positive_findings"],
        "uncertain": confidence_output["indeterminate_findings"],
        "normal": confidence_output["normal"]
    }
