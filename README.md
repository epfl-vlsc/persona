
# Persona

The Persona shell consists of a simple command parser and set of Python submodules that implement common bioinformatics tools using Persona's TensorFlow-based dataflow framework. 

Before continuing, make sure you have installed the [Persona System](https://github.com/epfl-vlsc/persona-system) and entered the virtual environment:

```shell
source path-to-persona-system/python-dev/bin/activate
```

The `persona` Python script is the entry point to the Persona environment. You may want to add it to your PATH. 
Use `persona -h` to get a list of all available functions.

First, you will probably want to import a dataset. Persona operates on the AGD format, which is efficient and allows for seamless scale-out distribution. 
Use one of the import functions to import your data, for example:

```shell
persona import-fastq --paired -p 4 --compress-parallel 6 -w 2 -o <output_directory> -n my_dataset file_1.fastq file_2.fastq
```

will import a paired FASTQ dataset into an AGD dataset called "my\_dataset". Use `persona import_fastq -h` to see what all the options are for. 
Generally, many will control parallelism at the IO read, compute, and IO write stages. 
You can tweak these numbers to get better performance. 

The `output_directory` will now contain your paired dataset in AGD format. 

To display ranges of records in the terminal you can use the display command

```shell
persona display 0 7 -d <dataset_directory> <dataset_directory>/metadata.json
```

This can be useful for sanity checking results. 
