import mysql.connector
import json
import csv


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
    metadata[aid] = {'order': ord_,
                     'family': fam,
                     'subfamily': sub,
                     'genus': gen,
                     'species': spc,
                     'baltimore': blt,
                     'organism': organism,
                     'sequence_length': sequence_lengths[aid]
                     }

  cnx.close()

  hosts = get_virus_hosts(signatures, metadata)
  for signature in signatures:
    metadata[signature]['hosts'] = hosts[metadata[signature]['species']]

  return metadata


def get_sequence_lengths(signatures, cnx):
  cursor = cnx.cursor()

  signature_aids = ', '.join(
      ["'{}'".format(signature) for signature in signatures])

  query = ("select aid, seq from genome where aid in ({})".format(signature_aids))

  cursor.execute(query)

  sequence_lengths = {}
  for (aid, seq) in cursor:
    sequence_lengths[aid] = len(seq)

  return sequence_lengths


def get_virus_hosts(signatures, metadata, file=None):
  """
    Finds the virus hosts a tab separated file.  Does not find all of the
    hosts since they apprently can have synonymous names.  Unlike the other
    functions here, it requires the species names of the viruses.
  """
  if file is None:
    # Found at http://www.genome.jp/virushostdb/
    # ftp://ftp.genome.jp/pub/db/virushostdb/virushostdb.tsv
    file = 'virushostdb.tsv'

  species = [metadata[signature]['species'] for signature in signatures]

  return _find_hosts(file, species)


def _find_hosts(file, species, host_class=False):
  hosts_list = {}
  with open(file) as f:
    rd = csv.reader(f, delimiter="\t", quotechar='"')
    for row in rd:
      match = matches(species, row)
      if not match is None:

        if host_class:
          host_split = row[9].split("; ")
          if len(host_split) > 4:
            host = host_split[4]
          else:
            host = ''
        else:
          host = row[8]

        if match in hosts_list and host not in hosts_list[match] and host != "":
          hosts_list[match] += [host]
        elif host != "":
          hosts_list[match] = [host]

  hosts = {}
  for spc in species:
    if spc in hosts_list:
      hosts[spc] = ", ".join(sorted(hosts_list[spc]))
    else:
      hosts[spc] = "Not Found"

  return hosts


def matches(species, row):
  row_species = row[1]
  for spc in species:
    if spc in row_species:
      return spc
    split_spc = spc.split(" ")

    # Greekify if herpes or endornavirus
    greeks = ['alpha', 'beta', 'gamma']
    if len(split_spc) > 1 and split_spc[1] == "herpesvirus":
      for greek in greeks:
        greek_spc = " ".join([split_spc[0], greek + split_spc[1], split_spc[2]])
        if greek_spc in row_species:
          return spc
    elif len(split_spc) > 2 and split_spc[2] == "endornavirus":
      for greek in greeks:
        greek_spc = " ".join([split_spc[0], split_spc[1], greek + " ".join(split_spc[2:])])
        if greek_spc in row_species:
          return spc

  return None


if __name__ == "__main__":
  metadata = get_metadata_for(['AB026117.1', 'AC_000005.1'])
  print(metadata)
