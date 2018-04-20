import mysql.connector
import json


def get_metadata_for(signatures):
  config = json.load(open('db_config.json'))
  cnx = mysql.connector.connect(**config)
  cursor = cnx.cursor()

  signature_aids = ', '.join(
      ["'{}'".format(signature) for signature in signatures])

  query = ("select aid, ord, fam, sub, gen, spc, organism from virus where aid in ({})".format(signature_aids))

  cursor.execute(query)

  metadata = {}
  for (aid, ord_, fam, sub, gen, spc, organism) in cursor:
    metadata[aid] = {'order': ord_, 'family': fam, 'subfamily': sub,
                     'genus': gen, 'species': spc, 'organism': organism}

  cnx.close()

  return metadata

if __name__ == "__main__":
  metadata = get_metadata_for(['AB026117.1', 'AC_000005.1'])
  print(metadata)
