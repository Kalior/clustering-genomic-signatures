#! /usr/bin/python3.6

import ftplib
import numpy as np
import os
import subprocess
import gzip
import shutil
from Bio import SeqIO, Entrez
import mysql.connector
import json


def download(organism, email, cnx, cursor):
  Entrez.email = email
  seach_terms = ' AND '.join(['{}[organism]'.format(organism),
                              'complete genome[All Fields]',
                              'biomol_genomic[PROP]',
                              '("35000"[SLEN] : "5000000"[SLEN])'])
  print(seach_terms)
  search_handle = Entrez.esearch(db='Nucleotide', retmax=100,
                                 term=seach_terms,
                                 idtype='acc')
  search_record = Entrez.read(search_handle)

  print('"(' + " OR ".join(["virus.aid='{}'".format(s) for s in search_record["IdList"]]) + ')"')

  fasta_handles = Entrez.efetch(db="Nucleotide", id=search_record["IdList"],
                                rettype="fasta", retmode="text")
  gb_handles = Entrez.efetch(db="Nucleotide", id=search_record["IdList"],
                             rettype="gb", retmode="text")

  fastas = list(SeqIO.parse(fasta_handles, "fasta"))
  gbs = list(SeqIO.parse(gb_handles, "genbank"))

  for i in range(len(fastas)):
    gb = gbs[i]
    fasta = fastas[i]
    to_db(gb, fasta, organism, cnx, cursor)


def to_db(gb, fasta, organism, cnx, cursor):
  organism_data = {
      'aid': gb.id,
      'ord': gb.annotations['taxonomy'][-3],
      'fam': gb.annotations['taxonomy'][-2],
      'sub': '',
      'gen': gb.annotations['taxonomy'][-1],
      'spc': gb.annotations['organism'],
      'dsc': gb.description,
      'blt': -1,
      'act': 1,
      'organism': organism
  }

  genome_data = {
      'aid': gb.id,
      'fasta': fasta.format('fasta'),
      'descr': gb.description,
      'seq': str(fasta.seq)
  }

  if not 'N' in str(fasta.seq) and not 'Y' in str(fasta.seq):
    try:
      add_to_db(cursor, organism_data, genome_data)
      # remove_from_db(cursor, gb.id)
      cnx.commit()
    except mysql.connector.errors.IntegrityError:
      print("{} already exists in db.".format(gb.id))


def add_to_db(cursor, organism_data, genome_data):
  add_organism = ("INSERT INTO virus "
                  "(aid, ord, fam, sub, gen, spc, dsc, blt, act, organism) "
                  "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
  add_genome = ("INSERT INTO genome (aid, fasta, descr, seq)"
                "VALUES(%s, %s, %s, %s)")
  cursor.execute(add_virus, tuple(organism_data.values()))
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
  download("plants", email, cnx, cursor)
  cnx.commit()
  cursor.close()
  cnx.close()

if __name__ == '__main__':
  main()
