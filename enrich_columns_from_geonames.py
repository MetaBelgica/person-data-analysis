import csv
import re
import logging
import utils
import requests
import os
from dotenv import load_dotenv
from csv_logger import CSVFileHandler
from argparse import ArgumentParser, Action

LOGGER_NAME="ENRICH_WITH_FIELDS_FROM_GEONAMES"
logger = logging.getLogger(LOGGER_NAME)


# -----------------------------------------------------------------------------
def main(inputFilename, outputFilename, geonamesIdColumn, columnFieldPairs, lang, apiUrl, logLevel='INFO', logFile=None):

  load_dotenv()
  setupLogging(logLevel, logFile)
  with open(inputFilename, 'r') as inFile, \
       open(outputFilename, 'w') as outFile:

    inputReader = csv.DictReader(inFile)

    outputFieldnames = inputReader.fieldnames + list(columnFieldPairs.values())
    outputWriter = csv.DictWriter(outFile, fieldnames=outputFieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    outputWriter.writeheader()

    # the cache dictionary in which we will store fetched countryIDs, e.g. {"BE": "2802361"}
    countryIDs = {}
    for row in inputReader:

      geonamesIdValue = row[geonamesIdColumn]
      geonamesIds = geonamesIdValue.split(';') if ';' in geonamesIdValue else [geonamesIdValue]

      valueList = []
      for geonamesId in geonamesIds:
        values = getGeoNamesFields(apiUrl, geonamesId, columnFieldPairs, lang)
        if values:
          valueList.append(values)

      if valueList:
        newRowData = { key: ';'.join(d[key] for d in valueList) for key in valueList[0]}
        row.update(newRowData)

      outputWriter.writerow(row)

# -----------------------------------------------------------------------------
def getGeoNamesFields(apiUrl, geonamesId, columnFieldPairs, lang):

  url = apiUrl.rstrip('/') + '/' + geonamesId
  verificationCertFilename = os.getenv('API_ROOT_CERTIFICATE')
  payload = {'fields': ','.join(columnFieldPairs.keys())}
  if lang:
    payload['lang'] = lang

  try:
    response = requests.get(url, verify=verificationCertFilename, params=payload)
    response.raise_for_status()

    data = response.json()
    return {columnFieldPairs[name]: val for name, val in data.items()}

  except requests.exceptions.HTTPError as e:
    logger.error(f'HTTPError for "{geonamesId}": {e}')
    return None
  except Exception as e:
    logger.error(f'API error requesting data for {geonamesId}": {e}')
    return None


  

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
class ParseDict(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest) or {}

        if values:
            for item in values:
                split_items = item.split("=", 1)
                key = split_items[
                    0
                ].strip()  # we remove blanks around keys, as is logical
                value = split_items[1]

                d[key] = value

        setattr(namespace, self.dest, d)



# -----------------------------------------------------------------------------
def parseArguments():

  parser = ArgumentParser(description='This script adds columns to a CSV based on an API call with a GeoNames ID')
  parser.add_argument('inputFile', help='The input file containing CSV records')
  parser.add_argument('-o', '--output-file', action='store', required=True, help='The output CSV file with new columns')
  parser.add_argument('-g', '--geonamesId-column', action='store', required=True, help='The name of the column in which the GeoNames ID is stored')
  parser.add_argument('-c', '--column-field-pairs', metavar='KEY=VALUE', nargs='+', action=ParseDict, help='Pairs where the key indicates the GeoNames field to extract and key the corresponding column name where it should be stored')
  parser.add_argument('--lang', action='store', help='Optional 2 letters iso code to indicate a language. Used to filter certain GeoNames fields such as name country.name, etc')
  parser.add_argument('--api-endpoint', action='store', required=True, help='The url to request the GeoNames info')
  parser.add_argument('-l', '--log-file', action='store', help='The optional name of the logfile')
  parser.add_argument('-L', '--log-level', action='store', default='INFO', help='The log level, default is INFO')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parseArguments()
  main(args.inputFile, args.output_file, args.geonamesId_column, args.column_field_pairs, args.lang, args.api_endpoint, logLevel=args.log_level, logFile=args.log_file)
