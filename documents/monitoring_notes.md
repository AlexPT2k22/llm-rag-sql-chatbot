# Field Monitoring Module

## Overview

The Field Monitoring module manages vineyard plot and property data, including geospatial information,
crop varieties, soil profiles, and irrigation systems.

## Entities and Properties

### Properties
Each agricultural entity manages one or more **properties** (farms). A property contains:
- Name and total area
- Location (district, municipality, parish)
- Associated agricultural year

### Plots
Each property contains **plots** (individual vineyard parcels). Plot attributes include:
- Area (total and usable)
- Planting year
- Row and inter-row spacing
- Slope and altitude
- Training system (e.g., Guyot, Cordon)
- Trellising type

### Crops and Varieties
Plots are associated with crops and grape varieties:
- Crop type (e.g., Vine, Olive)
- Grape variety (e.g., Touriga Nacional, Alvarinho)
- Production mode (e.g., Integrated Production, Organic)
- Planted area per variety

## Onde encontrar na aplicacao

1. Navigate to **Field Management** in the sidebar
2. Select **Plots** to view and manage all vineyard plots
3. Use the **Crop Registry** to assign varieties to plots
4. Check **Irrigation** for watering system configuration

## Key Views

The system provides pre-built views for common queries:
- **Field Overview**: Properties, plots, area, location, training systems
- **Crop-Variety**: Crops per plot with varieties and areas
