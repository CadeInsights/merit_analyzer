from dataclasses import fields


def dataclass_to_xml(data, n=1000):
    parts = []
    for field in fields(data):
        value = getattr(data, field.name)
        if value is None:
            continue
        parts.append(f"<{field.name}>{str(value)[:n]}</{field.name}>")
    return "".join(parts)
