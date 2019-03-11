import argparse
import codecs
import sys
import json


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')

def write_file(fname, data):
    with open(fname, 'w', encoding='utf-8') as outfile:
        return json.dump(data, reader(outfile), ensure_ascii=False, indent=2)
    
def read_file(fname):
    f = open(fname, 'r')
    data = json.load(f)
    f.close()
    return data


class classifier:

    def __init__(self):
        """
        Initializes a new data structure, which is a list of tuples to store data for classification.
        """
        self.list_data = {"authors":{}, "emails":{}, "description":"", "corpus":[]}
        


    def add_entry(self, data, label):
        """
        adds a new data entry to list of data
        """
        #tuple1=(data, label)
        self.list_data["corpus"].append({"data":data, "label":label})
        

    '''
    
    Add methods to gather_data which contains all data you will need for classification

    Add methods to do classification and evaluate it.
    
    '''
    

# the following command was used to produce sample.json from sample.data
# python3 group_project.py --file sample.json --authors "jon may" "sarik ghazarian" --description "knock knock jokes with binary classification by jon of whether the joke was 'funny' to a six year old or 'lame'" --readfile sample.data --emails jonmay@usc.edu sarik@usc.edu

def main():
    
    parser = argparse.ArgumentParser(description="Save gathered data into a json file. Read data from saved json file, make a classifier and then evaluate its performance.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", nargs='?', required=True, help="input/output file")
    parser.add_argument("--authors", nargs='+', required=True, help="authors' names")
    parser.add_argument("--emails", nargs='+', required=True, help="authors' email addresses")
    parser.add_argument("--description", nargs='+', required=True, help="project description")
    parser.add_argument("--readfile", nargs='?', default=None, help="tab separated data file that will be encoded into json and written (clobbers --file). if not present, assume we're writing")

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if args.readfile is not None:
        cls = classifier()
        cls.list_data["description"]=args.description
        for author in args.authors:
            acount = len(cls.list_data["authors"])+1
            cls.list_data["authors"]["author{}".format(acount)]=author
        for email in args.emails:
            acount = len(cls.list_data["emails"])+1
            cls.list_data["emails"]["email{}".format(acount)]=email

        f = open(args.readfile, 'r')
        for line in f:
            d = {}
            print(line)
            tup = line.strip().split('\t')
            print(tup[1])
            print(len(tup))
            cls.add_entry(*tup)

        outfile = write_file(args.file, cls.list_data)
    else:
        infile = read_file(args.file) 

    '''
    infile contains all data you need to do classification
    '''


if __name__ == '__main__':
    main()
