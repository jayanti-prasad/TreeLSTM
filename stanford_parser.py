import os
from argparse import Namespace


def get_params():
    cfg = Namespace (
       base_dir="data",
       lib_dir="lib",
       stanford_basedir="/Users/jayanti/Software/nlp-tools/stanford/",
       parser="stanford-parser-full-2018-10-17/stanford-parser.jar",
       postagger="stanford-postagger-full-2018-10-16/stanford-postagger.jar",
       model="stanford-english-corenlp-2018-10-05-models.jar"
      )

    return cfg 


def split_text(cfg, filepath, output_dir):
    output_dir = cfg.base_dir + os.sep + output_dir 
    if not os.path.exists(output_dir):
          os.makedirs(output_dir)

    print("split test output :", output_dir)  
    with open(filepath) as datafile, \
        open(os.path.join(output_dir, 'a.txt'), 'w') as afile, \
        open(os.path.join(output_dir, 'b.txt'), 'w') as bfile,  \
        open(os.path.join(output_dir, 'id.txt'), 'w') as idfile, \
        open(os.path.join(output_dir, 'sim.txt'), 'w') as simfile:
        datafile.readline()

        for line in datafile:
            i, a, b, sim, ent = line.strip().split('\t')
            idfile.write(i + '\n')
            afile.write(a + '\n')
            bfile.write(b + '\n')
            simfile.write(sim + '\n')
    print("wrote split_text") 



class StanfordParser:
    def __init__(self, cfg):
 
       self.cfg = cfg 
       parser = cfg.stanford_basedir + os.sep + cfg.parser
       postagger = cfg.stanford_basedir + os.sep + cfg.postagger
       model = cfg.stanford_basedir + os.sep + cfg.model
       class_path=[cfg.lib_dir, parser, postagger, model] 
       self.class_path = ":".join(class_path) 

    def compile_src(self):

       if not os.path.exists(cfg.lib_dir):
          os.makedirs(cfg.lib_dir)

       if not os.path.exists(cfg.lib_dir + os.sep + "ConstituencyParse.class"):
           print("Compiling java source files !")
           cmd = ('javac -cp %s lib/*.java -d %s' % (self.class_path, cfg.lib_dir))
           os.system(cmd)

       
    def dependency_parse(self, input_file, output_dir,  tokenize=True):
        print('\nDependency parsing ' + input_file)
        output_dir = self.cfg.base_dir + os.sep + output_dir

        prefix = os.path.splitext(os.path.basename(input_file))[0]
        tokpath = os.path.join(output_dir, prefix + '.toks')
        parentpath = os.path.join(output_dir, prefix + '.parents')
        relpath = os.path.join(output_dir, prefix + '.rels')

        tokenize_flag = '-tokenize - ' if tokenize else ''
        cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
               % (self.class_path, tokpath, parentpath, relpath, tokenize_flag, 
               output_dir + os.sep + input_file))
        os.system(cmd)
 
        print("dependency parsing done !")

    def constituency_parse(self, input_file, output_dir, tokenize=True):
        
        output_dir = self.cfg.base_dir + os.sep + output_dir

        prefix = os.path.splitext(os.path.basename(input_file))[0]
 
        tokpath = os.path.join(output_dir, prefix + '.toks')
        parentpath = os.path.join(output_dir, prefix + '.cparents')

        tokenize_flag = '-tokenize - ' if tokenize else ''
        cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
           % (self.class_path, tokpath, parentpath, tokenize_flag,  
           output_dir + os.sep + input_file))
        os.system(cmd)

        print("constituency  parsing done !")

if __name__ == "__main__":

    cfg = get_params()

    S = StanfordParser(cfg)
   
    # create class files if not present 
    S.compile_src()

    # read input and create split text 
    split_text(cfg, "sick_data/SICK_train.txt", "train")
    split_text(cfg, "sick_data/SICK_test_annotated.txt", "test")
    split_text(cfg, "sick_data/SICK_trial.txt", "dev")

    for output_dir  in ["train","test","dev"]:
        # create dependency for the first & second sentence
        S.dependency_parse("a.txt", output_dir)
        S.dependency_parse("b.txt", output_dir)

        # create constituency for the first & second sentence 
        S.constituency_parse("a.txt", output_dir)
        S.constituency_parse("b.txt", output_dir)

