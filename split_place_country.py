import csv
import re
import logging
import utils
from csv_logger import CSVFileHandler
from argparse import ArgumentParser

LOGGER_NAME="SPLIT_PLACE_COUNTRY"
logger = logging.getLogger(LOGGER_NAME)

# -----------------------------------------------------------------------------
def main(inputFilename, outputFilename, splitColumnName, placeColumnName, countryColumnName, otherColumns, delimiterPair=('(', ')'), logLevel='INFO', logFile=None):

  setupLogging(logLevel, logFile)
  with open(inputFilename, 'r') as inFile, \
       open(outputFilename, 'w') as outFile:

    inputReader = csv.DictReader(inFile)

    outputFieldnames = [placeColumnName, countryColumnName ] + otherColumns
    outputWriter = csv.DictWriter(outFile, fieldnames=outputFieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    outputWriter.writeheader()
    invalidCountryFormattingCounter = 0
    for row in inputReader:
      valueString = row[splitColumnName]
      # Get the country name or take the default country name from the config
      #
      try:
        if valueString != '':


          if utils.needs_encoding_fixing(valueString):
            valueString = utils.fix_encoding(valueString)

          # for KMSKB data
          placename = valueString.replace('[dut]','').strip()
          placename = placename.split(',')[0] if ',' in placename else placename
          placename = placename.split(' / ')[0] if ' / ' in placename else placename

          placename = getPlaceName(placename, delimiterPair=delimiterPair)
          countryName = getCountryName(valueString, delimiterPair=delimiterPair)

          outputRow = {k:v for k,v in row.items() if k in otherColumns}
          outputRow[placeColumnName] = placename
          outputRow[countryColumnName] = countryName
          outputWriter.writerow(outputRow)
 
      except Exception as e:
        logger.error('Could not split placename string into place and country ({valueString})')
        invalidCountryFormattingCounter += 1
        continue


# -----------------------------------------------------------------------------
def getPlaceName(placenameString, delimiterPair=('(', ')')):
  """Returns the name of a place in the given string.

  >>> getPlaceName('Gent')
  'Gent'
  >>> getPlaceName('Gent (Belgium)')
  'Gent'
  >>> getPlaceName('Gent (Belgium')
  'Gent'

  >>> getPlaceName('Gent [Belgium]', delimiterPair=('[', ']'))
  'Gent'
  """
  if delimiterPair[0] in placenameString or delimiterPair[1] in placenameString:
    components = placenameString.split(delimiterPair[0])
    if len(components) == 2:
      placename = components[0]
    elif len(components) == 1:
      placename = placenameString
    else: 
      msg = f'Multiple opening parentheses: {placenameString}'
      logger.warning(msg)
      raise Exception(msg)

    return placename.strip()
  return placenameString

# -----------------------------------------------------------------------------
def getCountryName(placenameString, delimiterPair=('(', ')')):
  """ Returns the name of the country in the given string.
      Use the nation, in case the more broad nation is given as well,
      for example England, United Kingdom

  >>> getCountryName("Gent")
  ''
  >>> getCountryName("Brussels (Belgium)")
  'Belgium'
  >>> getCountryName("Devon (Engeland, Verenigd Koninkrijk) [dut]")
  'Verenigd Koninkrijk'
  >>> getCountryName("Paris (France)")
  'France'

  >>> getCountryName("Paris [France]", delimiterPair=('[', ']'))
  'France'

  >>> getCountryName("Paris[FR]", delimiterPair=('[', ']'))
  'FR'

  Both Angleterre and England result in UK as the geonames API returns nothing for those values
  >>> getCountryName("Castor (Angleterre)")
  'UK'

  >>> getCountryName("Oxford (England)")
  'UK'

  >>> getCountryName("Paris (France")
  Traceback (most recent call last):
      ...
  Exception: Invalid country formatting "Paris (France"
  """
  if delimiterPair[0] in placenameString:
 
    pattern = r'.*' + '\\' + delimiterPair[0] + '(.*)' + '\\' + delimiterPair[1] + '.*'
    countryPattern = re.search(pattern, placenameString)

    if countryPattern:
      countryString = countryPattern.groups()[0]
    else:
      raise Exception(f'Invalid country formatting "{placenameString}"')

    # possibly an additional nation after the comma, let's take that one
    if ',' in countryString:
      country = countryString.split(',')[1].strip()
    elif countryString == 'Angleterre' or countryString == 'England':
      return 'UK'
    else:
      country = countryString
    return country
  else:
    return ""

# -----------------------------------------------------------------------------
def setupLogging(logLevel, logFile):

  logFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  if logFile:
    logger = logging.getLogger(LOGGER_NAME)
    # Debug: Print current handlers
    csvHandler = CSVFileHandler(logFile, logLevel=logLevel, delimiter=',', filemode='w')
    logger.addHandler(csvHandler)
  else:
    logging.basicConfig(level=logLevel, format=logFormat)
    logger = logging.getLogger(LOGGER_NAME)

# -----------------------------------------------------------------------------
def parseArguments():

  parser = ArgumentParser(description='This script splits a placename column')
  parser.add_argument('inputFile', help='The input file containing CSV records')
  parser.add_argument('-s', '--split-column', action='store', required=True, help='The name of the column containing placename and country')
  parser.add_argument('-p', '--placename-column', action='store', required=True, help='The name of the column in which the place name should be stored in the output CSV')
  parser.add_argument('-c', '--countryname-column', action='store', required=True, help='The name of the column in which the country name should be stored in the output CSV')
  parser.add_argument('-o', '--output-file', action='store', required=True, help='The output CSV file containing descriptive keys based on the key composition config')
  parser.add_argument('--other-column', action='append', required=True, help='Names of additional columns that should be added to the output')
  parser.add_argument('-d', '--delimiter-pair', nargs=2, metavar=('opening', 'closing'), default=('(', ')'), required=False, help='The opening and closing character for the country part of the string, e.g. ( and ) for Gent (Belgium) or [ and ] for Gent [Belgium]')
  parser.add_argument('-l', '--log-file', action='store', help='The optional name of the logfile')
  parser.add_argument('-L', '--log-level', action='store', default='INFO', help='The log level, default is INFO')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = parseArguments()
  main(args.inputFile, args.output_file, args.split_column, args.placename_column, args.countryname_column, args.other_column, delimiterPair=args.delimiter_pair, logLevel=args.log_level, logFile=args.log_file)
