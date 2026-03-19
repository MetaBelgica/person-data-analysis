# activate Python virtual environment
. ../../beltrans-data-integration/data-integration/py-integration-env/bin/activate

# make local Python modules available to be used via python -m tools.*
oldPythonPath=$PYTHONPATH
export PYTHONPATH=../../beltrans-data-integration

echo "### BIRTH PLACE"
python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_fr-BE_birth-places.csv \
  -c autID \
  -c birthPlace  \
  kik-irpa/kik-irpa-data-2026-nl-birthPlace.csv \
  kik-irpa/kik-irpa-data-2026-fr-birthPlace.csv

python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_nl-BE_birth-places.csv \
  -c autID \
  -c birthPlace  \
  kik-irpa/kik-irpa-data-2026-fr-birthPlace.csv \
  kik-irpa/kik-irpa-data-2026-nl-birthPlace.csv
echo ""

echo "### DEATH PLACE"
python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_fr-BE_death-places.csv \
  -c autID \
  -c deathPlace  \
  kik-irpa/kik-irpa-data-2026-nl-deathPlace.csv \
  kik-irpa/kik-irpa-data-2026-fr-deathPlace.csv

python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_nl-BE_death-places.csv \
  -c autID \
  -c deathPlace  \
  kik-irpa/kik-irpa-data-2026-fr-deathPlace.csv \
  kik-irpa/kik-irpa-data-2026-nl-deathPlace.csv

echo "### GENDER"
python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_fr-BE_gender.csv \
  -c autID \
  -c gender  \
  kik-irpa/kik-irpa-data-2026-nl-gender.csv \
  kik-irpa/kik-irpa-data-2026-fr-gender.csv

python -m tools.csv.difference2 \
  -o kik-irpa/2026-03-19_missing_nl-BE_gender.csv \
  -c autID \
  -c gender  \
  kik-irpa/kik-irpa-data-2026-fr-gender.csv \
  kik-irpa/kik-irpa-data-2026-nl-gender.csv
echo ""

deactivate
export PYTHONPATH=$oldPythonPath
