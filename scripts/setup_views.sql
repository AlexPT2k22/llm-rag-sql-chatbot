-- DB: Operations
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_field_base') THEN
    CREATE VIEW v_field_base AS
    SELECT
      p.gid               AS id_plot,
      p.denomination      AS plot_name,
      p.total_area,
      p.usable_area,
      p.avg_altitude,
      p.slope,
      p.planting_year,
      p.row_spacing,
      p.inter_row_spacing,
      ROUND((p.total_area / NULLIF(p.row_spacing * p.inter_row_spacing, 0))::numeric, 0)
                          AS estimated_plant_count,
      p.status            AS plot_status,
      prop.gid            AS id_property,
      prop.denomination   AS property_name,
      prop.id_entity,
      m.denomination      AS municipality,
      ts.denomination     AS training_system,
      tt.denomination     AS trellising_type,
      ti.denomination     AS irrigation_type,
      tr.denomination     AS rootstock,
      tc.denomination     AS conduction_system
    FROM plot p
    JOIN property prop ON prop.gid = p.id_property
    LEFT JOIN municipality m ON m.gid = prop.id_municipality
    LEFT JOIN type_training_system ts ON ts.gid = p.id_training_system
    LEFT JOIN type_trellising tt ON tt.gid = p.id_trellising
    LEFT JOIN plot_irrigation pti ON pti.id_plot = p.gid AND pti.status = 'A'
    LEFT JOIN type_irrigation ti ON ti.gid = pti.id_irrigation_type
    LEFT JOIN plot_rootstock ptr ON ptr.id_plot = p.gid AND ptr.status = 'A'
    LEFT JOIN type_rootstock tr ON tr.gid = ptr.id_rootstock
    LEFT JOIN plot_conduction ptc ON ptc.id_plot = p.gid AND ptc.status = 'A'
    LEFT JOIN type_conduction tc ON tc.gid = ptc.id_conduction
    WHERE p.status = 'A' AND prop.status = 'A';
    RAISE NOTICE 'View v_field_base created.';
  ELSE
    RAISE NOTICE 'View v_field_base already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_field_crop_variety') THEN
    CREATE VIEW v_field_crop_variety AS
    SELECT
      p.gid               AS id_plot,
      p.denomination      AS plot_name,
      p.total_area,
      prop.gid            AS id_property,
      prop.denomination   AS property_name,
      prop.id_entity,
      m.denomination      AS municipality,
      tc.denomination     AS crop_type,
      tv.denomination     AS variety,
      pv.area_crop        AS crop_area,
      tmp.denomination    AS production_mode
    FROM plot p
    JOIN property prop ON prop.gid = p.id_property
    LEFT JOIN municipality m ON m.gid = prop.id_municipality
    LEFT JOIN plot_variety pv ON pv.id_plot = p.gid AND pv.status = 'A'
    LEFT JOIN type_crop tc ON tc.gid = pv.id_crop_type
    LEFT JOIN type_variety tv ON tv.gid = pv.id_variety
    LEFT JOIN type_production_mode tmp ON tmp.gid = pv.id_production_mode
    WHERE p.status = 'A' AND prop.status = 'A';
    RAISE NOTICE 'View v_field_crop_variety created.';
  ELSE
    RAISE NOTICE 'View v_field_crop_variety already exists -- skipped.';
  END IF;
END $$;

-- BD: Plots
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_plot_resource_full') THEN
    CREATE VIEW v_plot_resource_full AS
    SELECT
      pc.gid                  AS id_practice,
      pc.start_date,
      pc.quantity_total_hours AS n_hours,
      tp.denomination         AS practice_type,
      p.denomination          AS plot_name,
      prop.denomination       AS property_name,
      prop.id_entity,
      pc.id_agricultural_year,
      tr.denomination         AS resource_type,
      r.denomination          AS resource,
      pcr.quantity,
      pcr.total_cost,
      tv.denomination         AS variety
    FROM cultural_practice pc
    JOIN practice_cultural_resource pcr ON pcr.id_practice = pc.gid
    JOIN resource r ON r.gid = pcr.id_resource
    JOIN type_resource tr ON tr.gid = r.id_resource_type
    JOIN plot p ON p.gid = pc.id_plot
    JOIN property prop ON prop.gid = p.id_property
    LEFT JOIN type_practice tp ON tp.gid = pc.id_practice_type
    LEFT JOIN plot_variety pv ON pv.id_plot = p.gid AND pv.status = 'A'
    LEFT JOIN type_variety tv ON tv.gid = pv.id_variety
    WHERE pc.status = 'A';
    RAISE NOTICE 'View v_plot_resource_full created.';
  ELSE
    RAISE NOTICE 'View v_plot_resource_full already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_matviews WHERE schemaname = 'public' AND matviewname = 'mv_plot_resource_full') THEN
    CREATE MATERIALIZED VIEW mv_plot_resource_full AS
    SELECT * FROM v_plot_resource_full;
    CREATE UNIQUE INDEX ON mv_plot_resource_full (id_practice, resource);
    RAISE NOTICE 'Materialized view mv_plot_resource_full created.';
  ELSE
    RAISE NOTICE 'Materialized view mv_plot_resource_full already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_matviews WHERE schemaname = 'public' AND matviewname = 'mv_calendar_hr') THEN
    CREATE MATERIALIZED VIEW mv_calendar_hr AS
    SELECT
      r.denomination AS resource,
      EXTRACT(MONTH FROM pc.start_date)::int AS month,
      EXTRACT(YEAR FROM pc.start_date)::int  AS year,
      SUM(pcr.quantity_total_hours)           AS total_hours,
      COUNT(DISTINCT pc.start_date::date)     AS days_worked,
      prop.id_entity
    FROM cultural_practice pc
    JOIN practice_cultural_resource pcr ON pcr.id_practice = pc.gid
    JOIN resource r ON r.gid = pcr.id_resource
    JOIN type_resource tr ON tr.gid = r.id_resource_type
    JOIN property prop ON prop.gid = p.id_property
    JOIN plot p ON p.gid = pc.id_plot
    WHERE tr.code = 'HR' AND pc.status = 'A'
    GROUP BY r.denomination, month, year, prop.id_entity;
    CREATE UNIQUE INDEX ON mv_calendar_hr (resource, year, month, id_entity);
    RAISE NOTICE 'Materialized view mv_calendar_hr created.';
  ELSE
    RAISE NOTICE 'Materialized view mv_calendar_hr already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_matviews WHERE schemaname = 'public' AND matviewname = 'mv_calendar_equipment') THEN
    CREATE MATERIALIZED VIEW mv_calendar_equipment AS
    SELECT
      r.denomination AS resource,
      EXTRACT(MONTH FROM pc.start_date)::int AS month,
      EXTRACT(YEAR FROM pc.start_date)::int  AS year,
      SUM(pcr.quantity_total_hours)           AS total_hours,
      COUNT(DISTINCT pc.start_date::date)     AS days_used,
      prop.id_entity
    FROM cultural_practice pc
    JOIN practice_cultural_resource pcr ON pcr.id_practice = pc.gid
    JOIN resource r ON r.gid = pcr.id_resource
    JOIN type_resource tr ON tr.gid = r.id_resource_type
    JOIN property prop ON prop.gid = p.id_property
    JOIN plot p ON p.gid = pc.id_plot
    WHERE tr.code = 'EQUIPMENT' AND pc.status = 'A'
    GROUP BY r.denomination, month, year, prop.id_entity;
    CREATE UNIQUE INDEX ON mv_calendar_equipment (resource, year, month, id_entity);
    RAISE NOTICE 'Materialized view mv_calendar_equipment created.';
  ELSE
    RAISE NOTICE 'Materialized view mv_calendar_equipment already exists -- skipped.';
  END IF;
END $$;

-- BD: Cellar
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_cellar_operations') THEN
    CREATE VIEW v_cellar_operations AS
    SELECT
      op.gid            AS id_operation,
      top.denomination  AS operation_type,
      op.registration_date,
      l.denomination    AS lot,
      t.denomination    AS tank,
      op.n_hours,
      op.total_cost,
      EXTRACT(YEAR FROM op.registration_date)::int AS year,
      ent.id_entity
    FROM cellar_operation op
    JOIN type_operation top ON top.gid = op.id_operation_type
    JOIN lot l ON l.gid = op.id_lot
    LEFT JOIN tank t ON t.gid = op.id_tank
    JOIN cellar_entity ent ON ent.gid = op.id_cellar_entity
    WHERE op.status = 'A';
    RAISE NOTICE 'View v_cellar_operations created.';
  ELSE
    RAISE NOTICE 'View v_cellar_operations already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_cellar_volumes') THEN
    CREATE VIEW v_cellar_volumes AS
    SELECT
      wt.denomination  AS wine_type,
      wp.denomination  AS phase,
      SUM(lc.estimated_volume) AS total_liters,
      ent.id_entity
    FROM lot_composition lc
    JOIN lot l ON l.gid = lc.id_lot
    JOIN wine w ON w.gid = l.id_wine
    JOIN wine_type wt ON wt.gid = w.id_wine_type
    JOIN wine_phase wp ON wp.gid = l.id_phase
    JOIN cellar_entity ent ON ent.gid = l.id_cellar_entity
    WHERE l.status = 'A' AND lc.status = 'A'
    GROUP BY wt.denomination, wp.denomination, ent.id_entity;
    RAISE NOTICE 'View v_cellar_volumes created.';
  ELSE
    RAISE NOTICE 'View v_cellar_volumes already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_cellar_losses') THEN
    CREATE VIEW v_cellar_losses AS
    SELECT
      tl.denomination  AS loss_type,
      l.denomination   AS lot,
      op.registration_date,
      op.n_hours       AS hours,
      op.total_cost,
      ent.id_entity
    FROM cellar_operation op
    JOIN type_loss tl ON tl.gid = op.id_loss_type
    JOIN lot l ON l.gid = op.id_lot
    JOIN cellar_entity ent ON ent.gid = op.id_cellar_entity
    WHERE op.id_loss_type IS NOT NULL AND op.status = 'A';
    RAISE NOTICE 'View v_cellar_losses created.';
  ELSE
    RAISE NOTICE 'View v_cellar_losses already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_cellar_consumables') THEN
    CREATE VIEW v_cellar_consumables AS
    SELECT
      c.denomination  AS consumable,
      SUM(co.quantity_out) AS total_quantity_out,
      AVG(co.avg_cost)     AS avg_unit_cost,
      ent.id_entity
    FROM consumable_operation co
    JOIN consumable c ON c.gid = co.id_consumable
    JOIN cellar_entity ent ON ent.gid = co.id_cellar_entity
    WHERE co.status = 'A'
    GROUP BY c.denomination, ent.id_entity;
    RAISE NOTICE 'View v_cellar_consumables created.';
  ELSE
    RAISE NOTICE 'View v_cellar_consumables already exists -- skipped.';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_views WHERE schemaname = 'public' AND viewname = 'v_cellar_hr_calendar') THEN
    CREATE VIEW v_cellar_hr_calendar AS
    SELECT
      r.denomination  AS worker_name,
      top.denomination AS operation_type,
      EXTRACT(MONTH FROM op.registration_date)::int AS month,
      EXTRACT(YEAR FROM op.registration_date)::int  AS year,
      SUM(opr.n_hours)     AS total_hours,
      SUM(opr.total_cost)  AS total_cost,
      ent.id_entity
    FROM cellar_operation op
    JOIN operation_worker opr ON opr.id_operation = op.gid
    JOIN resource r ON r.gid = opr.id_worker
    JOIN type_operation top ON top.gid = op.id_operation_type
    JOIN cellar_entity ent ON ent.gid = op.id_cellar_entity
    WHERE op.status = 'A'
    GROUP BY r.denomination, top.denomination, month, year, ent.id_entity;
    RAISE NOTICE 'View v_cellar_hr_calendar created.';
  ELSE
    RAISE NOTICE 'View v_cellar_hr_calendar already exists -- skipped.';
  END IF;
END $$;
