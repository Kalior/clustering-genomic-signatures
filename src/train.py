import os
import subprocess


def train(vlmcs, sequence_length, out_directory, number_of_parameters=128):
  os.system("rm -rf {}/*".format(out_directory))

  [write_sequence_as_fasta(vlmc, sequence_length, out_directory) for vlmc in vlmcs]

  list_path = os.path.join(out_directory, "list.txt")
  with open(list_path, 'w') as f:
    f.write('\n'.join([vlmc.name for vlmc in vlmcs]))

  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': number_of_parameters,
      'min_count': 4,
      'max_depth': 15,
      'count_free_parameters_individually': 'false',
      'generate_full_markov_chain': False,
      'markov_chain_order': 3
  }
  train_vlmcs(parameters, list_path, out_directory)


def write_sequence_as_fasta(vlmc, sequence_length, out_directory):
  vlmc.reset_sequence()
  size_of_line = 990
  context = vlmc.generate_sequence(50, 500)[-vlmc.order:]

  file_name = vlmc.name + ".fa"
  with open(os.path.join(out_directory, file_name), 'w') as f:
    f.write("> {}\n".format(vlmc.name))
    # Iteratively generate lines from a sequence of correct length
    for _ in range(sequence_length // size_of_line):
      sequence = vlmc.generate_sequence_from(size_of_line, context)
      f.write("{}\n".format(sequence))
      context = sequence[-vlmc.order]

    length_left = sequence_length % size_of_line
    sequence = vlmc.generate_sequence_from(length_left, context)
    f.write("{}\n".format(sequence))


def train_vlmcs(parameters, list_path, out_directory, input_directory=None, add_underlines_=True):
  if input_directory is None:
    input_directory = out_directory

  #model 0 = full order markov chain, 1 = variable markov chain
  if parameters['generate_full_markov_chain']:
    model_type = 0
  else:
    model_type = 1
  standard_args = "-pseudo -crr -f_f {} -ipf .fa -ipwd {} -opwd {} -osf TEST_ -m {} -frac 0 -revcomp".format(
      list_path, input_directory, out_directory, model_type)
  parameter_args = "-c_c {} -nc {} -npar {} -minc {} -kmax {} -free {} -k {}".format(
      parameters['use_constant_cutoff'],
      parameters['cutoff_value'],
      parameters['number_of_parameters'],
      parameters['min_count'],
      parameters['max_depth'],
      parameters['count_free_parameters_individually'],
      parameters['markov_chain_order']
  )

  args = ("../lib/classifier " + standard_args + " " + parameter_args).split()

  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  # print("Waited for classifier")
  # output = popen.stdout.read()
  # print(output.decode("utf-8"))
  if add_underlines_:
    # Needed for the parsing (since we expect them to be there...)
    add_underlines(out_directory)


def add_underlines(directory):
  for file_ in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    name, end = os.path.splitext(file_)
    orignal_name = os.path.join(directory, file_)
    new_name = os.path.join(directory, name + "__" + end)
    os.rename(orignal_name, new_name)
