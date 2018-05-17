#! /usr/bin/python3.6

# May need to set in mysql conf. (On my computer /etc/mysql/my.cnf:
# innodb_log_file_size = 256M

import ftplib
import numpy as np
import os
import subprocess
import gzip
import shutil
from Bio import SeqIO, Entrez
import mysql.connector
import json
import numpy as np


def download(organism, term, email, cnx, cursor, print_matches=False):
  Entrez.email = email
  search_handle = Entrez.esearch(db='Nucleotide', retmax=5000,
                                 term=term,
                                 idtype='acc')
  search_record = Entrez.read(search_handle)

  if print_matches:
    print('"(' + " OR ".join(["virus.aid='{}'".format(s) for s in search_record["IdList"]]) + ')"')

  batches = np.array_split(search_record["IdList"], 50)

  for batch in batches:
    if len(batch) <= 0:
      break
    batch = list(batch)
    fasta_handles = Entrez.efetch(db="Nucleotide", id=batch,
                                  rettype="fasta", retmode="text")
    gb_handles = Entrez.efetch(db="Nucleotide", id=batch,
                               rettype="gb", retmode="text")

    fastas = list(SeqIO.parse(fasta_handles, "fasta"))
    gbs = list(SeqIO.parse(gb_handles, "genbank"))
    # taxonomies = get_raw_taxonomy(gbs)

    for i in range(len(fastas)):
      gb = gbs[i]
      fasta = fastas[i]
      taxonomy = get_raw_taxonomy(gb)
      if taxonomy is not None:
        to_db(gb, fasta, taxonomy, organism, cnx, cursor)


def get_raw_taxonomy(gb):
  # It's very hard to get consistent taxonomy results.
  species = gb.annotations['organism']
  tax_handle = Entrez.esearch(db="Taxonomy", term=species, retmax=len(species))
  tax_search_record = Entrez.read(tax_handle)

  if len(tax_search_record['IdList']) < 1:
    print("{} not found in taxonomy".format(species))
    return None

  tax_fetch = Entrez.efetch(db="Taxonomy", id=tax_search_record['IdList'], retmode="xml")
  tax_record = list(Entrez.read(tax_fetch))
  taxonomies = [record['LineageEx'] for record in tax_record]
  return taxonomies[0]


def to_db(gb, fasta, taxonomy, organism, cnx, cursor):
  if len(taxonomy) < 3:
    return
  elif organism == 'virus':
    baltimore = parse_baltimore_class(taxonomy)
  else:
    baltimore = -1

  order, family, subfamily, genus = parse_taxonomy(taxonomy)

  # Filter out cases where we don't know the genus or family or baltimore
  if genus == '' or family == '' or (organism == 'virus' and baltimore == -1):
    print("Genus ({}), family ({}), or virus' Baltimore type ({}) not found.".format(
        genus, family, baltimore))
    return

  organism_data = {
      'aid': str(gb.id),
      'ord': str(order),
      'fam': str(family),
      'sub': str(subfamily),
      'gen': str(genus),
      'spc': str(gb.annotations['organism']),
      'dsc': str(gb.description),
      'blt': str(baltimore),
      'act': str(1),
      'organism': str(organism)
  }

  genome_data = {
      'aid': str(gb.id),
      'fasta': str(fasta.format('fasta')),
      'descr': str(gb.description),
      'seq': str(fasta.seq)
  }

  # Guard against this since Daniel's classifier can't handle them.
  # Another approach would be to replace them here, but seems to be
  # able to find sequences without ambiguities.
  ambiguous_alphabet = "UWSMKRYBDHVN"
  if not any(s in str(fasta.seq) for s in ambiguous_alphabet):
    try:
      print(organism_data)
      add_to_db(cursor, organism_data, genome_data)
      # remove_from_db(cursor, gb.id)
      # cnx.commit()
    except mysql.connector.errors.IntegrityError:
      print("{} already exists in db.".format(gb.id))


def parse_baltimore_class(taxonomy):
  primary_baltimore = taxonomy[1]['ScientificName']
  secondary_baltimore = taxonomy[2]['ScientificName']
  if primary_baltimore == 'dsDNA viruses, no RNA stage':
    return 1
  elif primary_baltimore == 'ssDNA viruses':
    return 2
  elif primary_baltimore == 'dsRNA viruses':
    return 3
  elif primary_baltimore == 'ssRNA viruses' and secondary_baltimore == 'ssRNA positive-strand viruses, no DNA stage':
    return 4
  elif primary_baltimore == 'ssRNA viruses' and secondary_baltimore == 'ssRNA negative-strand viruses':
    return 5
  elif primary_baltimore == 'Ortervirales':
    return 6
  elif primary_baltimore == 'Retro-transcribing viruses':
    return 7
  else:
    return -1


def parse_taxonomy(taxonomy):
  parsed_tax = {tax['Rank']: tax['ScientificName']
                for tax in taxonomy if tax['Rank'] in ['order', 'family', 'subfamily', 'genus']}

  return parsed_tax.get('order', ''), parsed_tax.get('family', ''), \
      parsed_tax.get('subfamily', ''), parsed_tax.get('genus', '')


def add_to_db(cursor, organism_data, genome_data):
  add_organism = ("INSERT INTO virus "
                  "(aid, ord, fam, sub, gen, spc, dsc, blt, act, organism) "
                  "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
  add_genome = ("INSERT INTO genome (aid, fasta, descr, seq)"
                "VALUES(%s, %s, %s, %s)")
  cursor.execute(add_organism, tuple(organism_data.values()))
  cursor.execute(add_genome, tuple(genome_data.values()))


def remove_from_db(cursor, id_):
  delete_organism = ("DELETE FROM virus WHERE aid=%s")
  delete_genome = ("DELETE FROM genome WHERE aid=%s")

  cursor.execute(delete_genome, (id_,))
  cursor.execute(delete_organism, (id_,))


def main():
  db_config = json.load(open('db_config.json'))
  cnx = mysql.connector.connect(**db_config)
  cursor = cnx.cursor()
  cursor.execute('set global max_allowed_packet=67108864')

  email = ''

  # seach_terms = ' AND '.join(['{}[organism]'.format(organism),
  #                             '"complete genome"[All Fields]',
  #                             'biomol_genomic[PROP]',
  #                             '("35000"[SLEN] : "5000000"[SLEN])'])
  # term = ('plants[organism] NOT "mitochondrion"[Title] NOT "plastid"[Title] NOT "mitochondrial"[Title] NOT "scaffold"[Title] NOT "chloroplast"[Title] NOT "shotgun"[Title] NOT "IN PROGRESS"[Title] NOT "UNVERIFIED"[Title] NOT "clone"[Title] AND "chromosome 1"[Title] NOT "contig"[Title] NOT "mRNA"[Title] NOT "partial sequence"[Title] NOT "cds"[Title] NOT "pseudogene"[Title] NOT "Predicted"[Title] AND cds[feature]')

  # print(seach_terms)

  term = 'viruses[organism] AND "complete genome"[title] AND ("18000"[SLEN] : "5000000"[SLEN])'
  download("virus", term, email, cnx, cursor)
  cnx.commit()
  cursor.close()
  cnx.close()

if __name__ == '__main__':
  main()
