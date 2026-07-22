# AgriSystem — Overview

## System Description

AgriSystem is an integrated agricultural management platform for vineyard operations.
It covers three core domains:

1. **Field Monitoring** — Property and plot management, crop varieties, geospatial data
2. **Field Operations** — Cultural practices, phytosanitary applications, fertilization, harvesting, resources
3. **Cellar Management** — Grape intake, winemaking operations, lot tracking, inventory, bottling

## Key Features

- **Plot Management**: Register and track vineyard plots with GPS coordinates, area, slope, training systems
- **Crop Registry**: Assign crop types and grape varieties to plots with production modes
- **Practice Tracking**: Log all field operations with resources, hours, and costs
- **Resource Management**: Manage workers, equipment, and agricultural products
- **Cellar Operations**: Track winemaking from grape intake to bottling
- **Inventory**: Monitor wine volumes, consumables, and losses
- **Cost Analysis**: Cost aggregation by property, plot, practice type, and resource

## Technical Architecture

The platform consists of:
- **Web application** for desktop use
- **Mobile companion** for in-field data entry
- **PostgreSQL database** with three schema domains
- **AI Chatbot** for natural language queries (this project)

## Database Structure

| Database | Domain |
|---|---|
| `operations` | Field monitoring: plots, properties, varieties, districts |
| `plots` | Field operations: practices, resources, harvesting, purchasing |
| `cellar` | Cellar management: lots, tanks, intake, bottling, inventory |

## AI Chatbot Integration

The AgriSystem AI Chatbot provides natural language access to:
- Documentation and procedures (RAG)
- Database queries (Text-to-SQL)
- Combined documentation + data responses

Ask questions like:
- "How do I register a new plot?"
- "What were the total hours worked on pruning in 2025?"
- "How much DOC Red wine is currently in stock?"
