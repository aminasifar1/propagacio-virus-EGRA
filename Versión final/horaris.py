import csv, json, os
from collections import defaultdict

DAY_ORDER = ["mon", "tue", "wed", "thu", "fri"]

def hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)

def minutes_to_slot(mins: int, day_start_mins: int, slot_minutes: int) -> int:
    return (mins - day_start_mins) // slot_minutes

def build_schedule_from_csv(
    csv_path: str,
    slot_minutes: int = 30,
    day_start: str = "08:00",
    travel_lead_slots: int = 1
) -> dict:
    day_start_mins = hhmm_to_minutes(day_start)

    # group -> day -> list[sessions]
    week = defaultdict(lambda: {d: [] for d in DAY_ORDER})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"].strip()
            if row["day"].isdigit():
                day = DAY_ORDER[int(row["day"])]
            else:
                day = row["day"].strip().lower()
            if row["start"].isdigit():
                start_slot = int(row["start"])
            else:
                start = hhmm_to_minutes(row["start"])
                start_slot = minutes_to_slot(start, day_start_mins, slot_minutes)
            if row["end"].isdigit():
                end_slot = int(row["end"])
            else:
                end = hhmm_to_minutes(row["end"])
                end_slot = minutes_to_slot(end, day_start_mins, slot_minutes)

            sess = {
                "start_slot": int(start_slot),
                "end_slot": int(end_slot),
                "room": row["room"].strip()
            }
            if "course" in row and row["course"].strip():
                sess["course"] = row["course"].strip()
            if "type" in row and row["type"].strip():
                sess["type"] = row["type"].strip()

            week[group][day].append(sess)

    # ordena sesiones por hora
    out = {}
    for group, days in week.items():
        for d in DAY_ORDER:
            days[d].sort(key=lambda s: s["start_slot"])
        out[group] = days
    return out

if __name__ == "__main__":
    data = build_schedule_from_csv(os.path.join(os.getcwd(),"Versión final","Horarios.csv"), slot_minutes=30, day_start="08:00", travel_lead_slots=1)
    with open("horaris_generated.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ Generado: horaris_generated.json")
