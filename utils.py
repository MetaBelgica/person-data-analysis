from datetime import datetime
import xml.etree.ElementTree as ET
import unicodedata as ud
import ast
import numpy as np
import enchant
import re
from stdnum import isbn
from stdnum import exceptions

# -----------------------------------------------------------------------------
def jsonParser(data):
    return ast.literal_eval(data) if data != "" else np.nan

# -----------------------------------------------------------------------------
def non_empty_percentage(series):
    non_na_series = series.dropna()
    return (non_na_series.astype(bool).sum() / len(series)) * 100

# -----------------------------------------------------------------------------
def getListOfIdentifiers(authorityID, rawString, identifierName, stats):
  """This function extracts and formats identifiers.

  >>> getListOfIdentifiers('1', '0000000000000001', 'ISNI', {})
  ['0000000000000001']
  >>> getListOfIdentifiers('1', '0000000000000001;0000 0000 0000 0002', 'ISNI', {})
  ['0000000000000001', '0000000000000002']
  >>> getListOfIdentifiers('1', '1234', 'VIAF', {})
  ['1234']
  """

  extractedIdentifiers = []
  if ';' in rawString:
    count(stats, f'more-than-one-{identifierName}')
    identifiers = rawString.split(';')
    for i in identifiers:
      if i != '':
        identifier = extractIdentifier(authorityID, f'{identifierName} {i}', pattern=identifierName)
        if identifier != '':
          extractedIdentifiers.append(identifier)
  else:
    if rawString != '':
      identifier = extractIdentifier(authorityID, f'{identifierName} {rawString}', pattern=identifierName)
      if identifier != '':
        extractedIdentifiers.append(identifier)

  return extractedIdentifiers
 
# -----------------------------------------------------------------------------
def determineISNIInconsistency(value):
  """This filter function returns True if there is more than one ISNI identifier in a set.

  False if there is only one ISNI to begin with
  >>> determineISNIInconsistency(set(['0000 0000 0000 0001']))
  False

  False if the same ISNI with different formatting
  >>> determineISNIInconsistency(set(['0000 0000 0000 0001', '0000000000000001']))
  False

  False if the same ISNI with different formatting
  >>> determineISNIInconsistency(set(['0000 0000 0000 0001\\n', '0000000000000001']))
  False

  True if different ISNIs
  >>> determineISNIInconsistency(set(['0000 0000 0000 0001', '0000000000000002']))
  True

  False if there isn't any ISNI
  >>> determineISNIInconsistency(set([]))
  False
  """
  if isinstance(value, set) and len(value) > 1:
    normalized = set(map(lambda x: x.strip().replace(' ', ''), value))
    if len(normalized) > 1:
      return True
  return False


# -----------------------------------------------------------------------------
def determineDateInconsistency(value):
  """This filter function returns True if two dates are different.
  """
  if isinstance(value, set) and len(value) > 1:
    return True  
  return False


# -----------------------------------------------------------------------------
def parseYear(year, patterns):
  """"This function returns a string representing a year based on the input and a list of possible patterns.

  >>> parseYear('2021', ['%Y'])
  '2021'
  >>> parseYear('2021', ['(%Y)', '%Y'])
  '2021'
  >>> parseYear('(2021)', ['%Y', '(%Y)'])
  '2021'
  """

  parsedYear = None
  for p in patterns:

    try:
      tmp = datetime.strptime(year, p).date().year
      parsedYear = str(tmp)
      break
    except ValueError:
      pass

  if parsedYear == None:
    return year
  else:
    return parsedYear


# -----------------------------------------------------------------------------
def compute_column_percentage(df, group_col, count_col, percentage_col):
    """
    Computes the percentage of the count of a specified column with respect to the count of another column
    after grouping by a specified column.

    Args:
        df (pd.DataFrame): The DataFrame to perform the operation on.
        group_col (str): The column to group by (e.g., 'dataSource').
        count_col (str): The column to count the total number of occurrences (e.g., 'autID').
        percentage_col (str): The column for which the percentage is computed (e.g., 'isni').

    Returns:
        pd.DataFrame: A DataFrame with the count of 'count_col', the count of 'percentage_col', 
                      and the computed percentage.
    """
    # Explode the 'percentage_col' if it contains lists
    #df_exploded = df.explode(percentage_col)
    df_exploded = df

    # Perform the aggregation: count occurrences for both 'count_col' and 'percentage_col'
    grouped_df = df_exploded.reset_index().groupby(group_col).agg({
        count_col: 'count',
        percentage_col: 'count'
    })

    # Calculate the percentage of the 'percentage_col' with respect to the 'count_col'
    grouped_df['percentage'] = (grouped_df[percentage_col] / grouped_df[count_col]) * 100

    # Format the percentage to two decimal places
    grouped_df['percentage'] = grouped_df['percentage'].round(2)

    return grouped_df

# -----------------------------------------------------------------------------
def parseDate(date, patterns):
  """"This function returns a string representing a date based on the input and a list of possible patterns.

  >>> parseDate('2021', ['%Y'])
  '2021'
  >>> parseDate('2021', ['(%Y)', '%Y'])
  '2021'
  >>> parseDate('(2021)', ['%Y', '(%Y)'])
  '2021'

  A correct date string for a correct input.
  >>> parseDate('1988-04-25', ['%Y-%m-%d'])
  '1988-04-25'

  A correct date string for dates with slash.
  >>> parseDate('25/04/1988', ['%Y-%m-%d', '%Y/%m/%d', '%Y/%m/%d', '%d/%m/%Y'])
  '1988-04-25'

  An empty value if the pattern is not found.
  >>> parseDate('25/04/1988', ['%Y-%m-%d', '%Y/%m/%d'])
  ''

  A correct date string for dates without delimiter.
  >>> parseDate('19880425', ['%Y-%m-%d', '%Y%m%d'])
  '1988-04-25'

  Only year and month are invalid.
  >>> parseDate('1988-04', ['%Y%m', '%Y-%m'])
  ''
  >>> parseDate('198804', ['%Y-%m', '%Y%m'])
  ''

  Keep year if this is the only provided information.
  >>> parseDate('1988', ['%Y-%m-%d', '%Y'])
  '1988'

  Keep year if it is in round or square brackets or has a trailing dot.
  >>> parseDate('[1988]', ['%Y', '[%Y]'])
  '1988'
  >>> parseDate('(1988)', ['(%Y)'])
  '1988'
  >>> parseDate('1988.', ['%Y', '%Y.'])
  '1988'


  """

  parsedDate = None
  for p in patterns:

    try:
      # try if the value is a year
      tmp = datetime.strptime(date, p).date()
      if len(date) == 4:
        parsedDate = str(tmp.year)
      elif len(date) > 4 and len(date) <= 7:
        if any(ele in date for ele in ['(', '[', ')', ']', '.']):
          parsedDate = str(tmp.year)
        else:
          parsedDate = ''
      else:
        parsedDate = str(tmp)
      break
    except ValueError:
      pass

  if parsedDate == None:
    return ''
  else:
    return parsedDate




# -----------------------------------------------------------------------------
def getDataprofileRecordFromMARCXML(elem, fieldMapping):
  """This function iterates over the MARC XML record given in 'elem' and creates a dictionary based on MARC fields and the provided 'fieldMapping'.

  >>> getDataprofileRecordFromMARCXML('', {})
  """
  for df in elem:
    if(df.tag == EQ.QName(NS_MARCSLIM, 'controlfield')):
      pass
    if(df.tag == EQ.QName(NS_MARCSLIM, 'datafield')):
      tagNumber = df.attrib['tag']
      if(tagNumber == '020'):
        pass
      elif(tagNumber == '041'):
        pass
      elif(tagNumber == '044'):
        pass
      elif(tagNumber == '245'):
        pass
      elif(tagNumber == '250'):
        pass
      elif(tagNumber == '264'):
        pass
      elif(tagNumber == '300'):
        pass
      elif(tagNumber == '700'):
        pass
      elif(tagNumber == '710'):
        pass
      elif(tagNumber == '765'):
        pass
      elif(tagNumber == '773'):
        pass
      elif(tagNumber == '775'):
        pass
      elif(tagNumber == '911'):
        pass
      elif(tagNumber == '944'):
        pass


# -----------------------------------------------------------------------------
def getElementValue(elem, sep=';'):
  """This function returns the value of the element if it is not None, otherwise an empty string.

  The function returns the 'text' value if there is one
  >>> class Test: text = 'hello'
  >>> obj = Test()
  >>> getElementValue(obj)
  'hello'

  It returns nothing if there is no text value
  >>> class Test: pass
  >>> obj = Test()
  >>> getElementValue(obj)
  ''

  And the function returns a semicolon separated list in case the argument is a list of objects with a 'text' attribute
  >>> class Test: text = 'hello'
  >>> obj1 = Test()
  >>> obj2 = Test()
  >>> getElementValue([obj1,obj2])
  'hello;hello'

  In case one of the list values is empty
  >>> class WithContent: text = 'hello'
  >>> class WithoutContent: text = None
  >>> obj1 = WithContent()
  >>> obj2 = WithoutContent()
  >>> getElementValue([obj1,obj2])
  'hello'
  """
  if elem is not None:
    if isinstance(elem, list):
      valueList = list()
      for e in elem:
        if hasattr(e, 'text'):
          if e.text is not None:
            valueList.append(e.text)
      return sep.join(valueList)
    else:
      if hasattr(elem, 'text'):
        return elem.text
  
  return ''


# -----------------------------------------------------------------------------
def extractNameComponents(value):
  """This function tries to extract a family name and a last name from the input and returns them as a tuple.

  >>> extractNameComponents('Lieber, Sven')
  ('Lieber', 'Sven')
  >>> extractNameComponents('van Gogh, Vincent')
  ('van Gogh', 'Vincent')

  Empty strings are returned if it did not work. If there is only one value, we assume the family name
  >>> extractNameComponents('')
  ('', '')
  >>> extractNameComponents('van Gogh')
  ('van Gogh', '')
  >>> extractNameComponents('Hermann')
  ('Hermann', '')
  """
  familyName = ''
  givenName = ''

  if value != '':
  
    components = value.split(',')
    if len(components) == 0:
      familyName = value
    elif len(components) == 1:
      familyName = components[0].strip()
    elif len(components) > 1:
      familyName = components[0].strip()
      givenName = components[1].strip()
 
  return (familyName, givenName) 


# -----------------------------------------------------------------------------
def extractIdentifier(rowID, value, pattern):
  """Extracts the digits of an identifier from 'value' based on the type of identifier ('value' starts with 'pattern').

  >>> extractIdentifier('1', 'ISNI 0000 0000 0000 1234', 'ISNI')
  '0000000000001234'
  >>> extractIdentifier('1', 'ISNI --', 'ISNI')
  ''
  >>> extractIdentifier('1', 'ISNI ?', 'ISNI')
  ''
  >>> extractIdentifier('1', 'VIAF --', 'VIAF')
  ''
  >>> extractIdentifier('1', '1234', 'ISNI')
  ''
  >>> extractIdentifier('1', 'VIAF 1234', 'VIAF')
  '1234'
  >>> extractIdentifier('1', '', 'ISNI')
  ''
  >>> extractIdentifier('1', 'ISNI 0000 0000 0000 1234 5678', 'ISNI')
  ''
  >>> extractIdentifier('1', 1234, 'ISNI')
  ''
  >>> extractIdentifier('1', "ISNI 0000 0001 2138 0055\\n", 'ISNI')
  '0000000121380055'
  >>> extractIdentifier('1', "ISNI \\n0000 0004 4150 6067", 'ISNI')
  '0000000441506067'
  """

  identifier = ''

  if( isinstance(value, str) ):

    value = value.strip()
    if(str.startswith(value, pattern) and not str.endswith(value, '-') and not '?' in value):
      # remove the prefix (e.g. VIAF or ISNI) and replace spaces (e.g. '0000 0000 1234')
      tmp = value.replace(pattern, '')
      #identifier = value.replace(pattern, '').replace(' ', '')
      tmp = tmp.strip()
      identifier = tmp.replace(' ', '')

      if(pattern == 'ISNI' and len(identifier) == 32):
        print("Several ISNI numbers (?) for '" + rowID + ": '" + identifier + "'")
        identifier = identifier[0:16]

  if pattern == 'ISNI':
    if len(identifier) != 16:
      return ''
    else:
      return str(identifier)
  else:
    return str(identifier)


# -----------------------------------------------------------------------------
def count(stats, counter, val=None, valueDict=None):
  """ This function simply adds to the given counter or creates it if not yet existing in 'stats'
      If the optional 'val' argument is given, the val is also logged in a set.

  >>> stats = {}
  >>> count(stats, 'myCounter')
  >>> stats['myCounter']
  1
  """
  if counter in stats:
    stats[counter] += 1
  else:
    stats[counter] = 1

  if val is not None:
      if counter in valueDict:
        valueDict[counter].add(val) 
      else:
        valueDict[counter] = set([val])



# -----------------------------------------------------------------------------
def compareStrings(s1, s2):
  """This function normalizes both strings and compares them. It returns true if it matches, false if not.

  >>> compareStrings("HeLlO", "Hello")
  True
  >>> compareStrings("judaïsme, islam, christianisme, ET sectes apparentées", "judaisme, islam, christianisme, et sectes apparentees")
  True
  >>> compareStrings("chamanisme, de l’Antiquité…)", "chamanisme, de lAntiquite...)")
  True
  """

  nS1 = getNormalizedString(s1)
  nS2 = getNormalizedString(s2)

  if(nS1 == nS2):
    return True
  else:
    return False

# -----------------------------------------------------------------------------
def getNormalizedString(s):
  """This function returns a normalized copy of the given string.

  >>> getNormalizedString("HeLlO")
  'hello'
  >>> getNormalizedString("judaïsme, islam, christianisme, ET sectes apparentées")
  'judaisme islam christianisme et sectes apparentees'
  >>> getNormalizedString("chamanisme, de l’Antiquité…)")
  'chamanisme de lantiquite)'

  >>> getNormalizedString("Abe Ce De ?")
  'abe ce de'
  >>> getNormalizedString("Abe Ce De !")
  'abe ce de'
  >>> getNormalizedString("Abe Ce De :")
  'abe ce de'

  >>> getNormalizedString("A. W. Bruna & zoon")
  'a w bruna & zoon'
  >>> getNormalizedString("A.W. Bruna & Zoon")
  'aw bruna & zoon'

  #>>> getNormalizedString("---")
  #''

  #>>> getNormalizedString("c----- leopard")
  #'c leopard'
  
  """
  charReplacements = {
    '.': '',
    ',': '',
    '?': '',
    '!': '',
    ':': '',
    ';': ''
  }

  # by the way: only after asci normalization the UTF character for ... becomes ...
  asciiNormalized = ud.normalize('NFKD', s).encode('ASCII', 'ignore').lower().strip().decode("utf-8")

  normalized = ''.join([charReplacements.get(char, char) for char in asciiNormalized])
  noDots = normalized.replace('...', '')
  # remove double whitespaces using trick from stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string
  return " ".join(noDots.split())
  
  
# -----------------------------------------------------------------------------
def createURIString(valueString, delimiter, vocab):
  """This function takes a delimiter separted string of values and returns a string
     in which every of these values is prefixed with the specified vocab URI.

  >>> createURIString('nl;fr;de', ';', 'http://id.loc.gov/vocabulary/languages/')
  'http://id.loc.gov/vocabulary/languages/nl;http://id.loc.gov/vocabulary/languages/fr;http://id.loc.gov/vocabulary/languages/de'

  An empty input string results in an empty output string
  >>> createURIString('', ';', 'http://id.loc.gov/vocabulary/languages/')
  ''

  Only a delimiter results in an empty string
  >>> createURIString(';', ';', 'http://id.loc.gov/vocabulary/languages/')
  ''

  """

  uris = []
  urisString = ""
  values = valueString.split(delimiter)
  if len(values) > 1: 
    for v in values:
      if len(v) > 0:
        uris.append(vocab + v) 
    urisString = ';'.join(uris)
  elif len(values) == 1:
    if len(values[0]) > 0:
      urisString = vocab + valueString
  else:
    urisString = ''

  return urisString

# -----------------------------------------------------------------------------
def preprocessISBNString(inputISBN):
  """This function normalizes a given string to return numbers only.

  >>> preprocessISBNString('978-90-8558-138-3 test')
  '9789085581383'
  >>> preprocessISBNString('9789085581383 test test')
  '9789085581383'
  >>> preprocessISBNString('9031411515')
  '9031411515'
  >>> preprocessISBNString('9791032305690')
  '9791032305690'
  >>> preprocessISBNString('978 90 448 3374')
  '978904483374'
  >>> preprocessISBNString('90 223 1348 4 (Manteau)')
  '9022313484'
  >>> preprocessISBNString('90 223 1348 4 (Manteau 123)')
  '9022313484'
  >>> preprocessISBNString('978-90-303-6744-4 (dl. 1)')
  '9789030367444'
  >>> preprocessISBNString('979-10-235-1393-613')
  '9791023513936'
  >>> preprocessISBNString('90-295-3453-2 (Deel 1)')
  '9029534532'
  >>> preprocessISBNString('I am not a ISBN number')
  ''
  >>> preprocessISBNString('')
  ''
  """

  inputISBNNorm = re.sub('\D', '', inputISBN)

  if len(inputISBNNorm) == 0:
    return ''
  elif len(inputISBNNorm) == 10:
    return inputISBNNorm
  elif len(inputISBNNorm) == 13:
    if inputISBNNorm.startswith('978') or inputISBNNorm.startswith('979'):
      return inputISBNNorm
    else:
      # it is a wrong ISBN number which happens to have 13 digits
      # Best shot: it probably is a 10 digit ISBN and there were other numbers as part of text
      return inputISBNNorm[:10]
  else:
    if len(inputISBNNorm) > 13:
      return inputISBNNorm[:13]
    elif len(inputISBNNorm) < 13 and len(inputISBNNorm) > 10:
      if inputISBNNorm.startswith('978') or inputISBNNorm.startswith('979'):
        # it is actually a wrong ISBN 13 number, nevertheless return all of it
        return inputISBNNorm
      else:
        # maybe number parts of the text got added by accident to a valid 10 digit ISBN
        return inputISBNNorm[:10]
    else:
      return inputISBNNorm


# -----------------------------------------------------------------------------
def getNormalizedISBN10(inputISBN):
  """This function normalizes an ISBN number.

  >>> getNormalizedISBN10('978-90-8558-138-3')
  '90-8558-138-9'
  >>> getNormalizedISBN10('978-90-8558-138-3 test')
  '90-8558-138-9'
  >>> getNormalizedISBN10('9789085581383')
  '90-8558-138-9'
  >>> getNormalizedISBN10('9031411515')
  '90-314-1151-5'
  >>> getNormalizedISBN10('9791032305690')
  ''
  >>> getNormalizedISBN10('')
  ''
  >>> getNormalizedISBN10('979-10-235-1393-613')
  ''
  >>> getNormalizedISBN10('978-10-235-1393-613')
  Traceback (most recent call last):
   ...
  stdnum.exceptions.InvalidFormat: Not a valid ISBN13.
  """

  inputISBNNorm = preprocessISBNString(inputISBN)

  if inputISBNNorm:
    isbn10 = None
    try:
      isbn10 = isbn.format(isbn.to_isbn10(inputISBNNorm))
      return isbn10
    except exceptions.InvalidComponent:
      # Probably an ISBN number with 979 prefix for which no ISBN10 can be created
      if inputISBNNorm.startswith('979'):
        return ''
      else:
        raise
  else:
    return ''

# -----------------------------------------------------------------------------
def getNormalizedISBN13(inputISBN):
  """This function normalizes an ISBN number.

  >>> getNormalizedISBN13('978-90-8558-138-3')
  '978-90-8558-138-3'
  >>> getNormalizedISBN13('978-90-8558-138-3 test')
  '978-90-8558-138-3'
  >>> getNormalizedISBN13('9789085581383')
  '978-90-8558-138-3'
  >>> getNormalizedISBN13('9031411515')
  '978-90-314-1151-1'
  >>> getNormalizedISBN13('')
  ''
  """

  inputISBNNorm = preprocessISBNString(inputISBN)

  if inputISBNNorm:
    isbn13 = None
    try:
      isbn13 = isbn.format(isbn.to_isbn13(inputISBNNorm))
      return isbn13
    except exceptions.InvalidFormat:
      print(f'Error in ISBN 13 conversion for "{inputISBN}"')
      raise
  else:
    return ''

# -----------------------------------------------------------------------------
def addDescriptiveKeyValue(value, configEntry, keyValues, prefixSuffix=''):
  """This function adds the value to found values based on some rules found in configEntry.

  >>> keys = set(['existingKey'])
  >>> addDescriptiveKeyValue("myKey",{"prefix": "myPrefix"}, keys)
  >>> sorted(keys)[1]
  'myPrefix/myKey'

  >>> addDescriptiveKeyValue("myKey2",{"prefix": "myPrefix"}, keys, prefixSuffix='Extra')
  >>> sorted(keys)[2]
  'myPrefixExtra/myKey2'
 
  """
  if value != '':
    if 'prefix' in configEntry:
      prefix = configEntry["prefix"] + prefixSuffix
      value = value.replace(' ','') if prefix == 'isni' else value
      keyValues.add(f'{prefix}/{value}')
    else:
      keyValues.add(f'{value}')

# -----------------------------------------------------------------------------
def needs_encoding_fixing(text):
    try:
        # Attempt a round-trip encode-decode cycle
        encoded = text.encode('latin1')
        decoded = encoded.decode('utf-8')
        # Return True if the decoded text looks different from the original
        return text != decoded
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Likely already correct UTF-8
        return False

# -----------------------------------------------------------------------------
def fix_encoding(text):
    try:
        # Decode from Latin-1 (or Windows-1252) and re-encode as UTF-8
        fixed_text = text.encode('latin1').decode('utf-8')
        return fixed_text
    except UnicodeDecodeError:
        # Return the original text if decoding fails
        return text




# -----------------------------------------------------------------------------
if __name__ == "__main__":
  import doctest
  doctest.testmod()
