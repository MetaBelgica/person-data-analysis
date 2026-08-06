
#
# KMSKB
#
echo "## KMSKB"
echo ""
#python enrich_columns_from_geonames.py \
#  kmskb/enriched/250909_kmskb-data-birthPlace-enriched.csv \
#  -o kmskb/enriched/birth_places_label_enriched.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

#python enrich_columns_from_geonames.py \
#  kmskb/enriched/250909_kmskb-data-deathPlace-enriched.csv \
#  -o kmskb/enriched/death_places_label_enriched.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName


#
# KIK-IRPA
#
echo "## KIK-IRPA"
echo ""
#python enrich_columns_from_geonames.py \
#  kik-irpa/enriched/Birth_place_CORR_KIK_IRPA.csv \
#  -o kik-irpa/enriched/birth_places_label_enriched.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

#python enrich_columns_from_geonames.py \
#  kik-irpa/enriched/Death_place_CORR_KIK_IRPA.csv \
#  -o kik-irpa/enriched/death_places_label_enriched.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

python enrich_columns_from_geonames.py \
  kik-irpa/enriched/reconciled_birth-place.csv \
  -o kik-irpa/enriched/reconciled_birth-place_enriched.csv \
  -g correctIDs \
  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
  --lang EN \
  -c country.code=countryCode name=correctIDsName

python enrich_columns_from_geonames.py \
  kik-irpa/enriched/reconciled_death-place.csv \
  -o kik-irpa/enriched/reconciled_death-place_enriched.csv \
  -g correctIDs \
  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
  --lang EN \
  -c country.code=countryCode name=correctIDsName



python enrich_columns_from_geonames.py \
  kik-irpa/enriched/2026-05-06_birth-place_benoit.csv \
  -o kik-irpa/enriched/2026-05-06_birth-place_benoit_enriched.csv \
  -g correctIDs \
  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
  --lang EN \
  -c country.code=countryCode name=correctIDsName

python enrich_columns_from_geonames.py \
  kik-irpa/enriched/2026-05-06_death-place_benoit.csv \
  -o kik-irpa/enriched/2026-05-06_death-place_benoit_enriched.csv \
  -g correctIDs \
  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
  --lang EN \
  -c country.code=countryCode name=correctIDsName


#
# KMKG
#
echo "## KMKG"
echo ""
#python enrich_columns_from_geonames.py \
#  "kmkg/enriched/kmkg-data-birthPlace-enriched_Multiple API results_Nacha_Sven.csv" \
#  -o kmkg/enriched/birth_places_label_enriched_Nacha.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

#python enrich_columns_from_geonames.py \
#  "kmkg/enriched/kmkg-data-birthPlace-enriched_No API response_Heloise_Sven.csv" \
#  -o kmkg/enriched/birth_places_label_enriched_Heloise.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName


#python enrich_columns_from_geonames.py \
#  "kmkg/enriched/kmkg-data-deathPlace-enriched_Multiple API results_Angelique_Sven.csv" \
#  -o kmkg/enriched/death_places_label_enriched_Angelique.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

#python enrich_columns_from_geonames.py \
#  "kmkg/enriched/kmkg-data-deathPlace-No API response_Marianne.csv" \
#  -o kmkg/enriched/death_places_label_enriched_Marianne.csv \
#  -g correctIDs \
#  --api-endpoint https://beltrans2.kbr.be/geonames-lookup/place \
#  --lang EN \
#  -c country.code=countryCode name=correctIDsName

