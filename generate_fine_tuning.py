"""
Generates fine_tuning_dataset.json with synthetic Q&A examples in ShareGPT format
for fine-tuning Qwen2.5 7B with Unsloth on the agricultural domain.
"""
import json

EXAMPLES = [
    # Operations — Properties and Plots
    ("What crops exist and what are their areas across entity properties?",
     "SELECT property_name, plot_name, crop_type, variety, crop_area\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id}\nORDER BY property_name, crop_type, variety\nLIMIT 100"),

    ("What is the total vineyard area in the municipality of Vila Real?",
     "SELECT crop_type, SUM(crop_area) AS total_area\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND crop_type ILIKE '%Vine%' AND municipality ILIKE '%Vila Real%'\nGROUP BY crop_type\nLIMIT 50"),

    ("What is the area of Touriga Nacional plots?",
     "SELECT property_name, plot_name, crop_type, variety, crop_area\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND variety ILIKE '%Touriga Nacional%'\nORDER BY crop_area DESC NULLS LAST\nLIMIT 50"),

    ("What is the vineyard area under Integrated Production?",
     "SELECT property_name, plot_name, crop_type, variety, crop_area, production_mode\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND production_mode ILIKE '%Integrated Production%'\nORDER BY crop_area DESC NULLS LAST\nLIMIT 50"),

    ("List grape varieties planted and area occupied by each",
     "SELECT variety, SUM(crop_area) AS total_area, COUNT(DISTINCT plot_name) AS n_plots\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND variety IS NOT NULL\nGROUP BY variety\nORDER BY total_area DESC NULLS LAST\nLIMIT 50"),

    ("What is the total vineyard area of the entity?",
     "SELECT SUM(crop_area) AS total_vineyard_area\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND crop_type ILIKE '%Vine%'\nLIMIT 10"),

    ("What is the total area by crop type?",
     "SELECT crop_type, SUM(crop_area) AS total_area, COUNT(DISTINCT plot_name) AS n_plots\nFROM v_field_crop_variety\nWHERE id_entity = {entity_id} AND crop_type IS NOT NULL\nGROUP BY crop_type\nORDER BY total_area DESC NULLS LAST\nLIMIT 50"),

    ("List plots with rootstock 110 Richter",
     "SELECT property_name, plot_name, rootstock, total_area\nFROM v_field_base\nWHERE rootstock ILIKE '%110 Richter%' AND id_entity = {entity_id}\nORDER BY total_area DESC NULLS LAST\nLIMIT 50"),

    # Plots — Cultural Practices
    ("What human resources worked on cultural practices and how many hours?",
     "SELECT resource, practice_type, SUM(n_hours) AS total_hours, COUNT(*) AS n_practices\nFROM mv_plot_resource_full\nWHERE resource_type = 'Cultural Practice HR' AND id_entity = {entity_id}\nGROUP BY resource, practice_type\nORDER BY total_hours DESC NULLS LAST\nLIMIT 50"),

    ("What equipment was used in practices and how many hours?",
     "SELECT resource, practice_type, SUM(n_hours) AS total_hours\nFROM mv_plot_resource_full\nWHERE resource_type = 'Cultural Practice Equipment' AND id_entity = {entity_id}\nGROUP BY resource, practice_type\nORDER BY total_hours DESC NULLS LAST\nLIMIT 50"),

    ("List phytosanitary applications during 2025",
     "SELECT practice_type, resource, plot_name, property_name, quantity, total_cost, start_date\nFROM mv_plot_resource_full\nWHERE resource_type = 'Cultural Practice Phyto' AND id_entity = {entity_id} AND EXTRACT(YEAR FROM start_date) = 2025\nORDER BY start_date DESC NULLS LAST\nLIMIT 50"),

    ("Where was AGRO-STAR applied during 2025?",
     "SELECT resource, plot_name, property_name, quantity, total_cost, start_date\nFROM mv_plot_resource_full\nWHERE resource_type = 'Cultural Practice Phyto' AND resource ILIKE '%AGRO-STAR%' AND id_entity = {entity_id} AND EXTRACT(YEAR FROM start_date) = 2025\nORDER BY start_date DESC NULLS LAST\nLIMIT 50"),

    ("Monthly calendar with hours worked by agricultural workers",
     "SELECT resource, month, year, total_hours, days_worked\nFROM mv_calendar_hr\nWHERE id_entity = {entity_id} AND year = {year}\nORDER BY year DESC NULLS LAST, month, resource\nLIMIT 50"),

    ("List days worked by Worker A during 2025 grouped by month",
     "SELECT resource, month, year, total_hours, days_worked\nFROM mv_calendar_hr\nWHERE resource ILIKE '%Worker A%' AND id_entity = {entity_id} AND year = 2025\nORDER BY month\nLIMIT 50"),

    ("List pruning practices with costs and hours grouped by variety",
     "SELECT practice_type, variety, plot_name, SUM(n_hours) AS total_hours, SUM(total_cost) AS total_cost\nFROM mv_plot_resource_full\nWHERE practice_type ILIKE '%Pruning%' AND id_entity = {entity_id}\nGROUP BY practice_type, variety, plot_name\nORDER BY total_hours DESC NULLS LAST\nLIMIT 50"),

    # Cross-Reference
    ("Show hours worked and costs by practice type for {year}",
     "SELECT practice_type, SUM(n_hours) AS total_hours, SUM(total_cost) AS total_cost\nFROM mv_plot_resource_full\nWHERE id_entity = {entity_id} AND EXTRACT(YEAR FROM start_date) = {year}\nGROUP BY practice_type\nORDER BY total_hours DESC NULLS LAST\nLIMIT 50"),

    ("Show costs and hours by resource type for each property",
     "SELECT property_name, resource_type, SUM(n_hours) AS total_hours, SUM(total_cost) AS total_cost\nFROM mv_plot_resource_full\nWHERE id_entity = {entity_id}\nGROUP BY property_name, resource_type\nORDER BY property_name, resource_type\nLIMIT 50"),

    # Cellar
    ("List all operations for lot DOC Red",
     "SELECT operation_type, lot, tank, registration_date, n_hours, total_cost\nFROM v_cellar_operations\nWHERE lot ILIKE '%DOC Red%' AND id_entity = {entity_id}\nORDER BY registration_date DESC NULLS LAST\nLIMIT 50"),

    ("Calculate costs and hours for all cellar lots in {year}",
     "SELECT lot, SUM(n_hours) AS total_hours, SUM(total_cost) AS total_cost\nFROM v_cellar_operations\nWHERE id_entity = {entity_id} AND year = {year}\nGROUP BY lot\nORDER BY total_cost DESC NULLS LAST\nLIMIT 50"),

    ("List wine quantities by wine type in the cellar",
     "SELECT wine_type, SUM(total_liters) AS total_liters\nFROM v_cellar_volumes\nWHERE id_entity = {entity_id}\nGROUP BY wine_type\nORDER BY total_liters DESC NULLS LAST\nLIMIT 50"),
]

# Expand with system prompt template
SYSTEM_PROMPT = """You are an expert SQL assistant for an agricultural management platform.
Generate valid PostgreSQL queries based on the user's natural language question.
Use only the columns available in the provided schema. Never invent columns."""

dataset = []
for question, sql in EXAMPLES:
    dataset.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql},
        ]
    })

with open("fine_tuning_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Generated {len(dataset)} fine-tuning examples.")
