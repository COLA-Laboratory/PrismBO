from xml.etree import ElementTree as ET
from typing import List, Dict, Union

# Parser function
def parse_task(xml_str: str) -> Dict[str, Union[str, List, Dict]]:
    root = ET.fromstring(xml_str)
    task = {
        "name": root.findtext("name").strip('"'),
        "description": root.findtext("desc").strip('"'),
        "objectives": [],
        "variables": [],
        "constraints": [],
        "statistics": {}
    }

    for obj in root.find("objectives"):
        name, direction = obj.text.strip().rsplit(" ", 1)
        task["objectives"].append({
            "name": name.strip('"'),
            "direction": direction
        })

    for var in root.find("variables"):
        parts = var.text.strip().split(" ", 2)
        var_name = parts[0].strip('"')
        var_type = parts[1]
        var_range = parts[2]
        if var_type == "categorical":
            values = var_range.strip("{}").replace('"', '').split(", ")
            var_range = values
        else:
            var_range = list(map(float, var_range.strip("[]").split(",")))
        task["variables"].append({
            "name": var_name,
            "type": var_type,
            "range": var_range
        })

    constraints_elem = root.find("constraints")
    if constraints_elem is not None:
        for con in constraints_elem:
            name, con_type = con.text.strip().rsplit(" ", 1)
            task["constraints"].append({
                "name": name.strip('"'),
                "type": con_type
            })

    stats_elem = root.find("stats")
    if stats_elem is not None:
        mean = stats_elem.findtext("mean")
        variance = stats_elem.findtext("variance")
        if mean is not None:
            task["statistics"]["mean"] = float(mean)
        if variance is not None:
            task["statistics"]["variance"] = float(variance)

    return task



if __name__ == "__main__":
    xml_input = """
    <task>
    <name>"Speed Reducer Design"</name>
    <desc>"A mechanical design task with one objective and inequality constraints on strength and size."</desc>

    <objectives>
        <obj>"weight" min</obj>
    </objectives>

    <variables>
        <var>"face_width" real [2.6, 3.6]</var>
        <var>"teeth_number" integer [17, 28]</var>
        <var>"material_type" categorical {"steel", "aluminum", "titanium"}</var>
    </variables>

    <constraints>
        <con>"stress_limit" ineq</con>
        <con>"gear_ratio_fix" eq</con>
    </constraints>

    <stats>
        <mean>295.3</mean>
        <variance>25.1</variance>
    </stats>
    </task>
    """
    parsed_task = parse_task(xml_input)
    print(parsed_task)