def parse_report(report_text: str):
    sections = {
        "summary": "",
        "findings": [],
        "impression": []
    }

    current = None

    for line in report_text.splitlines():
        line = line.strip()

        if line.lower().startswith("summary"):
            current = "summary"
            continue
        elif line.lower().startswith("findings"):
            current = "findings"
            continue
        elif line.lower().startswith("impression"):
            current = "impression"
            continue

        if not line:
            continue

        if current == "summary":
            sections["summary"] += line + " "
        elif current in ["findings", "impression"] and line.startswith("-"):
            sections[current].append(line[1:].strip())

    # Clean up
    sections["summary"] = sections["summary"].strip()
    sections["findings"] = sections["findings"][:5]
    sections["impression"] = sections["impression"][:5]

    return sections
