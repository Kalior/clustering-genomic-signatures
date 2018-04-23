import mysql.connector
import json


def get_metadata_for(signatures):
  config = json.load(open('db_config.json'))
  cnx = mysql.connector.connect(**config)

  sequence_lengths = get_sequence_lengths(signatures, cnx)
  cursor = cnx.cursor()

  signature_aids = ', '.join(
      ["'{}'".format(signature) for signature in signatures])

  query = ("select aid, ord, fam, sub, gen, spc, blt, organism from virus where aid in ({})".format(
      signature_aids))

  cursor.execute(query)

  metadata = {}
  for (aid, ord_, fam, sub, gen, spc, blt, organism) in cursor:
    metadata[aid] = {'order': ord_, 'family': fam, 'subfamily': sub,
                     'genus': gen, 'species': spc, 'baltimore': blt, 'organism': organism,
                     'sequence_length': sequence_lengths[aid]}

  cnx.close()

  return metadata


def get_sequence_lengths(signatures, cnx):
  cursor = cnx.cursor()

  signature_aids = ', '.join(
      ["'{}'".format(signature) for signature in signatures])

  query = ("select aid, seq from genome where aid in ({})".format(
      signature_aids))

  cursor.execute(query)

  sequence_lengths = {}
  for (aid, seq) in cursor:
    sequence_lengths[aid] = len(seq)

  return sequence_lengths

if __name__ == "__main__":
  metadata = get_metadata_for(['AB026117.1', 'AC_000005.1'])
  print(metadata)
