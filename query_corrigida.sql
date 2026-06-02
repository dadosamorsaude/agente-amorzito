{{ config(
  s3_data_dir = get_external_location("pdgt_amorsaude_tecnologia", this.name),
  format = 'parquet',
  table_type = 'hive',
  s3_data_naming = 'unique',
  materialized = 'table',
  schema = 'pdgt_amorsaude_tecnologia'
) }}

with cpf_pacientes_amorsaude as (
  select cpf_norm,
    id_amei,
    id_webdental
  from {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }}
  where webvidas_cpf is not null
    or webdental_cpf is not null
    or amei_cpf is not null
),
cpf_valido AS (
  select cpf_norm,
    COALESCE(
      CASE WHEN webvidas_cpf_valido = 1 THEN webvidas_cpf END,
      CASE WHEN webdental_cpf_valido = 1 THEN webdental_cpf END,
      CASE WHEN amei_cpf_valido = 1 THEN amei_cpf END
    ) AS cpf_final
  FROM {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }}
),
email_valido AS (
  select cpf_norm,
    COALESCE(
      CASE WHEN app_email_valido = 1 THEN app_email END,
      CASE WHEN webvidas_email_valido = 1 THEN webvidas_email END,
      CASE WHEN amei_email_valido = 1 THEN amei_email END,
      CASE WHEN webdental_email_valido = 1 THEN webdental_email END,
      CASE WHEN feegow_email_valido = 1 THEN feegow_email END,
      CASE WHEN cdt_email_valido = 1 THEN cdt_email END
    ) AS email_final
  FROM {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }}
),
celular_valido as (
  select cpf_norm,
    COALESCE(
      CASE WHEN app_celular_valido = 1 THEN app_celular END,
      CASE WHEN amei_celular_valido = 1 THEN amei_celular END,
      CASE WHEN webvidas_celular_valido = 1 THEN webvidas_celular END,
      CASE WHEN webdental_celular_valido = 1 THEN webdental_celular END,
      CASE WHEN feegow_celular_valido = 1 THEN feegow_celular END,
      CASE WHEN cdt_celular_valido = 1 THEN cdt_celular END,
      CASE WHEN cvortex_celular_valido = 1 THEN cvortex_celular END
    ) AS celular_final
  FROM {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }}
),
nome_final as (
  select cpf_norm,
    CASE
      WHEN max_len = LENGTH(TRIM(COALESCE(app_nome, ''))) AND app_nome IS NOT NULL THEN app_nome
      WHEN max_len = LENGTH(TRIM(COALESCE(amei_nome, ''))) AND amei_nome IS NOT NULL THEN amei_nome
      WHEN max_len = LENGTH(TRIM(COALESCE(feegow_nome, ''))) AND feegow_nome IS NOT NULL THEN feegow_nome
      WHEN max_len = LENGTH(TRIM(COALESCE(webdental_nome, ''))) AND webdental_nome IS NOT NULL THEN webdental_nome
      WHEN max_len = LENGTH(TRIM(COALESCE(cdt_nome, ''))) AND cdt_nome IS NOT NULL THEN cdt_nome
      WHEN max_len = LENGTH(TRIM(COALESCE(cvortex_nome, ''))) AND cvortex_nome IS NOT NULL THEN cvortex_nome
      ELSE COALESCE(webvidas_nome, '')
    END AS nome_paciente
  from (
    select *,
      GREATEST(
        LENGTH(TRIM(COALESCE(app_nome, ''))),
        LENGTH(TRIM(COALESCE(amei_nome, ''))),
        LENGTH(TRIM(COALESCE(feegow_nome, ''))),
        LENGTH(TRIM(COALESCE(webdental_nome, ''))),
        LENGTH(TRIM(COALESCE(cdt_nome, ''))),
        LENGTH(TRIM(COALESCE(cvortex_nome, ''))),
        LENGTH(TRIM(COALESCE(webvidas_nome, '')))
      ) as max_len
    from {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }}
  ) sub
),
email_alternativo AS (
  SELECT pu.cpf_norm,
    COALESCE(
      CASE WHEN app_email_valido = 1 AND lower(app_email) <> lower(ev.email_final) THEN app_email END,
      CASE WHEN webvidas_email_valido = 1 AND lower(webvidas_email) <> lower(ev.email_final) THEN webvidas_email END,
      CASE WHEN amei_email_valido = 1 AND lower(amei_email) <> lower(ev.email_final) THEN amei_email END,
      CASE WHEN webdental_email_valido = 1 AND lower(webdental_email) <> lower(ev.email_final) THEN webdental_email END,
      CASE WHEN feegow_email_valido = 1 AND lower(feegow_email) <> lower(ev.email_final) THEN feegow_email END,
      CASE WHEN cdt_email_valido = 1 AND lower(cdt_email) <> lower(ev.email_final) THEN cdt_email END
    ) as email_alternativo
  from {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }} pu
    left join email_valido ev on pu.cpf_norm = ev.cpf_norm
),
email_alternativo_2 AS (
  SELECT pu.cpf_norm,
    COALESCE(
      CASE WHEN app_email_valido = 1 AND lower(app_email) <> lower(ev.email_alternativo) AND lower(app_email) <> lower(ev2.email_final) THEN app_email END,
      CASE WHEN webvidas_email_valido = 1 AND lower(webvidas_email) <> lower(ev.email_alternativo) AND lower(webvidas_email) <> lower(ev2.email_final) THEN webvidas_email END,
      CASE WHEN amei_email_valido = 1 AND lower(amei_email) <> lower(ev.email_alternativo) AND lower(amei_email) <> lower(ev2.email_final) THEN amei_email END,
      CASE WHEN webdental_email_valido = 1 AND lower(webdental_email) <> lower(ev.email_alternativo) AND lower(webdental_email) <> lower(ev2.email_final) THEN webdental_email END,
      CASE WHEN feegow_email_valido = 1 AND lower(feegow_email) <> lower(ev.email_alternativo) AND lower(feegow_email) <> lower(ev2.email_final) THEN feegow_email END,
      CASE WHEN cdt_email_valido = 1 AND lower(cdt_email) <> lower(ev.email_alternativo) AND lower(cdt_email) <> lower(ev2.email_final) THEN cdt_email END
    ) as email_alternativo_2
  from {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }} pu
    left join email_alternativo ev on pu.cpf_norm = ev.cpf_norm
    left join email_valido ev2 on pu.cpf_norm = ev2.cpf_norm
),
celular_alternativo as (
  select pu.cpf_norm,
    COALESCE(
      CASE WHEN app_celular_valido = 1 and pu.app_celular <> cv.celular_final THEN app_celular END,
      CASE WHEN amei_celular_valido = 1 and pu.amei_celular <> cv.celular_final THEN amei_celular END,
      CASE WHEN webvidas_celular_valido = 1 and pu.webvidas_celular <> cv.celular_final THEN webvidas_celular END,
      CASE WHEN webdental_celular_valido = 1 and pu.webdental_celular <> cv.celular_final THEN webdental_celular END,
      CASE WHEN feegow_celular_valido = 1 and pu.feegow_celular <> cv.celular_final THEN feegow_celular END,
      CASE WHEN cdt_celular_valido = 1 and pu.cdt_celular <> cv.celular_final then pu.cdt_celular END,
      CASE WHEN cvortex_celular_valido = 1 and pu.cvortex_celular <> cv.celular_final THEN cvortex_celular END
    ) AS celular_alternativo
  FROM {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }} pu
    left join celular_valido cv on pu.cpf_norm = cv.cpf_norm
),
celular_alternativo_2 as (
  select pu.cpf_norm,
    COALESCE(
      CASE WHEN app_celular_valido = 1 and app_celular <> cal.celular_alternativo and app_celular <> cv.celular_final THEN app_celular END,
      CASE WHEN amei_celular_valido = 1 and pu.amei_celular <> cal.celular_alternativo and amei_celular <> cv.celular_final THEN amei_celular END,
      CASE WHEN webvidas_celular_valido = 1 and pu.webvidas_celular <> cal.celular_alternativo and webvidas_celular <> cv.celular_final THEN webvidas_celular END,
      CASE WHEN webdental_celular_valido = 1 and pu.webdental_celular <> cal.celular_alternativo and webdental_celular <> cv.celular_final THEN webdental_celular END,
      CASE WHEN feegow_celular_valido = 1 and pu.feegow_celular <> cal.celular_alternativo and feegow_celular <> cv.celular_final THEN feegow_celular END,
      CASE WHEN cdt_celular_valido = 1 and pu.cdt_celular <> cal.celular_alternativo and cdt_celular <> cv.celular_final then pu.cdt_celular END,
      CASE WHEN cvortex_celular_valido = 1 and pu.cvortex_celular <> cal.celular_alternativo and cvortex_celular <> cv.celular_final THEN cvortex_celular END
    ) AS celular_alternativo_2
  FROM {{ source('pdgt_amorsaude_tecnologia', 'fl_data_pacientes_unificados') }} pu
    left join celular_alternativo cal on pu.cpf_norm = cal.cpf_norm
    left join celular_valido cv on pu.cpf_norm = cv.cpf_norm
),
base_final as (
  select pas.cpf_norm,
    nf.nome_paciente,
    cpfv.cpf_final,
    ev.email_final,
    eal.email_alternativo,
    eal3.email_alternativo_2,
    cf.celular_final,
    cal.celular_alternativo,
    cal2.celular_alternativo_2
  from cpf_pacientes_amorsaude pas
    left join cpf_valido cpfv on pas.cpf_norm = cpfv.cpf_norm
    left join email_valido ev on pas.cpf_norm = ev.cpf_norm
    left join email_alternativo eal on pas.cpf_norm = eal.cpf_norm
    left join email_alternativo_2 eal3 on pas.cpf_norm = eal3.cpf_norm
    left join nome_final nf on pas.cpf_norm = nf.cpf_norm
    left join celular_valido cf on pas.cpf_norm = cf.cpf_norm
    left join celular_alternativo cal on pas.cpf_norm = cal.cpf_norm
    left join celular_alternativo_2 cal2 on pas.cpf_norm = cal2.cpf_norm
),
resultado as (
  SELECT
    cpf_norm as chave,
    id_amei,
    id_webdental,
    lower(nome_paciente) as nome_paciente,
    CASE
      WHEN length(regexp_replace(cpf_final, '[^0-9]', '')) = 11
        THEN regexp_replace(
               regexp_replace(cpf_final, '[^0-9]', ''),
               '([0-9]{3})([0-9]{3})([0-9]{3})([0-9]{2})',
               '$1.$2.$3-$4'
             )
      ELSE NULL
    END AS cpf_final,
    lower(email_final) as email_final,
    lower(email_alternativo) as email_alternativo,
    lower(email_alternativo_2) as email_alternativo_2,
    CASE
      WHEN length(regexp_replace(regexp_replace(celular_final, '[^0-9]', ''), '^55', '')) = 11
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_final, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{5})([0-9]{4})',
               '($1) $2-$3'
             )
      WHEN length(regexp_replace(regexp_replace(celular_final, '[^0-9]', ''), '^55', '')) = 10
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_final, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{4})([0-9]{4})',
               '($1) $2-$3'
             )
      ELSE NULL
    END AS celular_final,
    CASE
      WHEN length(regexp_replace(regexp_replace(celular_alternativo, '[^0-9]', ''), '^55', '')) = 11
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_alternativo, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{5})([0-9]{4})',
               '($1) $2-$3'
             )
      WHEN length(regexp_replace(regexp_replace(celular_alternativo, '[^0-9]', ''), '^55', '')) = 10
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_alternativo, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{4})([0-9]{4})',
               '($1) $2-$3'
             )
      ELSE NULL
    END AS celular_alternativo,
    CASE
      WHEN length(regexp_replace(regexp_replace(celular_alternativo_2, '[^0-9]', ''), '^55', '')) = 11
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_alternativo_2, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{5})([0-9]{4})',
               '($1) $2-$3'
             )
      WHEN length(regexp_replace(regexp_replace(celular_alternativo_2, '[^0-9]', ''), '^55', '')) = 10
        THEN regexp_replace(
               regexp_replace(regexp_replace(celular_alternativo_2, '[^0-9]', ''), '^55', ''),
               '([0-9]{2})([0-9]{4})([0-9]{4})',
               '($1) $2-$3'
             )
      ELSE NULL
    END AS celular_alternativo_2
  from base_final
)
SELECT *
FROM resultado
