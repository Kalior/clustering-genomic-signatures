import mysql.connector

config = {
    'user': 'virus',
    'password': 'virus',
    'host': 'localhost',
    'database': 'virus',
    'raise_on_warnings': True,
}


def get_metadata_for(signatures):
  cnx = mysql.connector.connect(**config)
  cursor = cnx.cursor()

  signature_aids = ', '.join(
      ["'{}'".format(signature) for signature in signatures])

  query = ("select aid, fam, gen, spc from virus where aid in ({})".format(signature_aids))

  cursor.execute(query)

  metadata = {}
  for (aid, fam, gen, spc) in cursor:
    metadata[aid] = {'family': fam, 'genus': gen, 'species': spc}

  cnx.close()

  return metadata

if __name__ == "__main__":
  metadata = get_metadata_for(['AB026117.1', 'AC_000005.1'])
  print(metadata)
