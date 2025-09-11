import csv
import re
import logging
import utils
import requests
import os
from dotenv import load_dotenv
from csv_logger import CSVFileHandler
from argparse import ArgumentParser

LOGGER_NAME="ENRICH_GEONAMES_COUNTRY"
logger = logging.getLogger(LOGGER_NAME)

COLUMN_BIRTH_COUNTRY_GEONAMES = "geonameIDCountryBirth"
COLUMN_DEATH_COUNTRY_GEONAMES = "geonameIDCountryDeath"

# -----------------------------------------------------------------------------
def main(inputFilename, outputFilename, countryCodeBirthColumn, countryCodeDeathColumn, apiUrl, logLevel='INFO', logFile=None):

  load_dotenv()
  setupLogging(logLevel, logFile)
  with open(inputFilename, 'r') as inFile, \
       open(outputFilename, 'w') as outFile:

    inputReader = csv.DictReader(inFile)

    outputFieldnames = inputReader.fieldnames + [COLUMN_BIRTH_COUNTRY_GEONAMES, COLUMN_DEATH_COUNTRY_GEONAMES]
    outputWriter = csv.DictWriter(outFile, fieldnames=outputFieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    outputWriter.writeheader()

    # the cache dictionary in which we will store fetched countryIDs, e.g. {"BE": "2802361"}
    countryIDs = {}
    for row in inputReader:

      # read country string
      countryCodeBirth = row[countryCodeBirthColumn]
      countryCodeDeath = row[countryCodeDeathColumn]
      # lookup country geonameID (in API or cache)
      row[COLUMN_BIRTH_COUNTRY_GEONAMES] = getGeonamesIDCountry(apiUrl, countryCodeBirth, countryIDs)
      row[COLUMN_DEATH_COUNTRY_GEONAMES] = getGeonamesIDCountry(apiUrl, countryCodeDeath, countryIDs)

      outputWriter.writerow(row)

# -----------------------------------------------------------------------------
def getGeonamesIDCountry(apiUrl, countryCode, countryIDs):
  if countryCode == '':
    return ''

  if countryCode in countryIDs:
    return countryIDs[countryCode]
  else:
    try:
      url = apiUrl.rstrip('/') + '/' + countryCode
      verificationCertFilename = os.getenv('API_ROOT_CERTIFICATE')
      response = requests.get(url, verify=verificationCertFilename)
      response.raise_for_status()
      geonamesID = response.text
      countryIDs[countryCode] = geonamesID
      return geonamesID

    except requests.exceptions.HTTPError as e:
      logger.error(f'none or too many search results for {countryCode}')
      return ''
    except Exception as e:
      logger.error(f'API error requesting country ID for "{countryCode}": {e}')
      return ''

# -----------------------------------------------------------------------------
def setupLogging(logLevel, logFile):

  logFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  if logFile:
    logger = logging.getLogger(LOGGER_NAME)
    csvHandler = CSVFileHandler(logFile, logLevel=logLevel, delimiter=',', filemode='w')
    logger.addHandler(csvHandler)
  else:
    logging.basicConfig(level=logLevel, format=logFormat)
    logger = logging.getLogger(LOGGER_NAME)

# -----------------------------------------------------------------------------
def parseArguments():

  parser = ArgumentParser(description='This script splits a placename column')
  parser.add_argument('inputFile', help='The input file containing CSV records')
  parser.add_argument('-o', '--output-file', action='store', required=True, help='The output CSV file containing descriptive keys based on the key composition config')
  parser.add_argument('--country-code-column-birth', action='store', required=True, help='The name of the column in which the country code of the birth place is stored')
  parser.add_argument('--country-code-column-death', action='store', required=True, help='The name of the column in which the country code of the death place is stored')
  parser.add_argument('--api-endpoint', action='store', required=True, help='The url to request the geonamesId of a given ISO 3166 country code')
  parser.add_argument('-l', '--log-file', action='store', help='The optional name of the logfile')
  parser.add_argument('-L', '--log-level', action='store', default='INFO', help='The log level, default is INFO')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parseArguments()
  main(args.inputFile, args.output_file, args.country_code_column_birth, args.country_code_column_death, args.api_endpoint, logLevel=args.log_level, logFile=args.log_file)
